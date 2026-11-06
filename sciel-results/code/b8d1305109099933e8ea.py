import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def is_valid_sequence(seq):
        """Check if all consecutive block products are distinct."""
        if not seq:
            return True
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True
    
    def fast_greedy():
        """Fast greedy: include x if it doesn't cause product collision."""
        values = []
        products = set()
        for x in range(2, 129):
            # Calculate new suffix products if we append x
            tails = []
            prod = x
            valid = True
            for a in reversed(values):
                prod *= a
                if prod in products:
                    valid = False
                    break
                tails.append(prod)
            
            if not valid:
                continue
            
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        
        return values
    
    def try_insert(value, seq):
        """Try to insert value into seq at all valid positions, return best extension."""
        # Try inserting at each position
        best = None
        for pos in range(len(seq) + 1):
            # Check if value can be inserted at pos while maintaining strictly increasing
            if pos > 0 and seq[pos-1] >= value:
                continue
            if pos < len(seq) and seq[pos] <= value:
                continue
            
            new_seq = seq[:pos] + [value] + seq[pos:]
            if is_valid_sequence(new_seq):
                if best is None or len(new_seq) > len(best):
                    best = new_seq
        return best
    
    def local_search_improve(seq, time_limit):
        """Try to improve seq by removing and reinserting elements."""
        best = list(seq)
        best_len = len(best)
        start = time.time()
        
        # Try removing each element and reinserting it elsewhere
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            for i in range(len(best)):
                if time.time() > start_time + deadline - 5:
                    return best
                # Remove element at i
                removed = best[i]
                candidate = best[:i] + best[i+1:]
                
                # Try reinserting removed at various positions
                for pos in range(len(candidate) + 1):
                    if pos > 0 and candidate[pos-1] >= removed:
                        continue
                    if pos < len(candidate) and candidate[pos] <= removed:
                        continue
                    
                    new_seq = candidate[:pos] + [removed] + candidate[pos:]
                    if is_valid_sequence(new_seq):
                        if len(new_seq) > best_len:
                            best = new_seq
                            best_len = len(best)
                            improved = True
                            break
                        # Even if same length, we might want to try different arrangements
                        # but for now, only accept if strictly better
                if improved:
                    break
        
        return best
    
    # Phase 1: Fast greedy baseline
    best_seq = fast_greedy()
    best_len = len(best_seq)
    
    # Phase 2: Beam search with proper state preservation
    # State: (seq_list, prods_set, sufs_list)
    # Key fix: if no state can be extended by x, keep the beam unchanged
    
    beam = [([], set(), [])]
    
    for x in range(2, 129):
        if time.time() > deadline - 10:  # Leave time for local search
            break
            
        new_beam = []
        for seq, prods, suffixes in beam:
            # Calculate new suffixes
            new_suffixes = [x]
            for s in suffixes:
                new_suffixes.append(x * s)
            
            new_prods = set(new_suffixes)
            
            # Check collision
            if prods.intersection(new_prods):
                continue
            
            # Valid extension
            new_seq = seq + [x]
            new_all_prods = prods.union(new_prods)
            
            new_beam.append((new_seq, new_all_prods, new_suffixes))
        
        # Update best_seq from current beam (these are valid sequences ending at < x)
        for s, _, _ in beam:
            if len(s) > best_len:
                best_len = len(s)
                best_seq = s
        
        if new_beam:
            # Dynamic beam width: wider early, narrower later
            progress = (x - 2) / 127
            if progress < 0.3:
                K = 80
            elif progress < 0.6:
                K = 50
            else:
                K = 30
            
            # Score by potential: length + bonus for smaller product set (more room)
            def potential_score(item):
                seq, prods, suffixes = item
                length = len(seq)
                prod_set_size = len(prods)
                last_elem = seq[-1] if seq else 2
                return (length, -prod_set_size, -last_elem)
            
            new_beam.sort(key=potential_score, reverse=True)
            # Keep top K states
            beam = new_beam[:K]
        # else: beam stays the same (states can potentially be extended by later values)
    
    # Final check from beam
    for s, _, _ in beam:
        if len(s) > best_len:
            best_len = len(s)
            best_seq = s
    
    # Phase 3: Local search improvement
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_seq = local_search_improve(best_seq, min(remaining_time - 2, 30))
        best_len = len(best_seq)
    
    # Phase 4: Random restarts to try to find better solutions
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Try a few random greedy variations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Randomized greedy: skip some values randomly
            seq = []
            products = set()
            skip_prob = 0.1 + attempt * 0.05
            for x in range(2, 129):
                if time.time() > deadline - 5:
                    break
                if random.random() < skip_prob:
                    continue
                
                tails = []
                prod = x
                valid = True
                for a in reversed(seq):
                    prod *= a
                    if prod in products:
                        valid = False
                        break
                    tails.append(prod)
                
                if not valid:
                    continue
                
                if not products.intersection(tails):
                    seq.append(x)
                    products.update(tails)
            
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq
    
    return best_seq
