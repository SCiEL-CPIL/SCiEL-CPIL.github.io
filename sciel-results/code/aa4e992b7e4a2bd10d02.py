import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin (evaluator kills at 90s)
    
    # Phase 1: Greedy baseline
    def greedy_construct():
        seq = []
        prods = set()
        for c in range(2, 129):
            # Check if adding c creates any collision
            tails = [c]
            prod = c
            for a in reversed(seq):
                prod *= a
                tails.append(prod)
            
            has_collision = False
            for p in tails:
                if p in prods:
                    has_collision = True
                    break
            
            if not has_collision:
                seq.append(c)
                prods.update(tails)
        
        return seq, prods
    
    best_seq, best_prods = greedy_construct()
    best_len = len(best_seq)
    
    if time.time() > deadline - 5:
        return best_seq
    
    # Phase 2: Beam search to improve
    BEAM_WIDTH = 150
    MAX_CANDIDATES = 15  # Try next 15 numbers after last
    
    # Initialize beams with variations of the greedy solution
    beams = [(list(best_seq), set(best_prods))]
    
    # Also add some random starts
    random.seed(42)
    for _ in range(20):
        rand_seq = []
        rand_prods = set()
        # Random subset with increasing values
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        for c in candidates:
            if rand_seq and c <= rand_seq[-1]:
                continue
            tails = [c]
            prod = c
            for a in reversed(rand_seq):
                prod *= a
                tails.append(prod)
            
            has_collision = any(p in rand_prods for p in tails)
            if not has_collision:
                rand_seq.append(c)
                rand_prods.update(tails)
        
        if len(rand_seq) >= best_len - 2:  # Only keep competitive starts
            beams.append((rand_seq, rand_prods))
    
    while time.time() < deadline - 3:
        new_beams = []
        
        for seq, prods in beams:
            if not seq:
                last_val = 1
            else:
                last_val = seq[-1]
                
            # Try candidates: focus on numbers close to last_val for density
            # but also allow some jumps
            start_c = last_val + 1
            end_c = min(last_val + MAX_CANDIDATES + 1, 129)
            
            for c in range(start_c, end_c):
                # Quick collision check
                tails = [c]
                prod = c
                for a in reversed(seq):
                    prod *= a
                    tails.append(prod)
                
                has_collision = False
                for p in tails:
                    if p in prods:
                        has_collision = True
                        break
                
                if not has_collision:
                    new_seq = seq + [c]
                    new_prods = prods.union(tails)
                    new_beams.append((new_seq, new_prods))
                    
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
        
        if not new_beams:
            break
        
        # Sort and prune beams
        # Key: length (desc), then last value (asc) for more room,
        # then use a hash of products for diversity
        def beam_sort_key(b):
            seq, prods = b
            l = -len(seq)
            last = seq[-1] if seq else 0
            # Add diversity: use sum of smallest products as tiebreaker
            # Smaller products = more room for future extensions
            if prods:
                min_prod = min(prods)
            else:
                min_prod = 0
            return (l, last, min_prod)
        
        new_beams.sort(key=beam_sort_key)
        
        # Keep top BEAM_WIDTH, but ensure diversity
        # Take top 70% by score, then sample from remaining for diversity
        keep_count = min(BEAM_WIDTH, len(new_beams))
        if keep_count > 100:
            top_count = int(keep_count * 0.7)
            top_beams = new_beams[:top_count]
            remaining = new_beams[top_count:keep_count]
            if remaining:
                # Sample some from remaining for diversity
                extra_count = min(keep_count - top_count, len(remaining))
                sampled = random.sample(remaining, extra_count)
                beams = top_beams + sampled
            else:
                beams = top_beams
        else:
            beams = new_beams[:keep_count]
        
        # If best sequence can't be extended, try to repair by inserting
        # numbers in gaps (local search)
        if time.time() < deadline - 5 and len(best_seq) < 100:
            # Try inserting numbers in gaps of best_seq
            improved = False
            for _ in range(10):  # Limit repair attempts
                if time.time() > deadline - 5:
                    break
                # Find a gap where we can insert
                best_seq_sorted = sorted(best_seq)
                for i in range(len(best_seq_sorted) - 1):
                    gap_start = best_seq_sorted[i]
                    gap_end = best_seq_sorted[i + 1]
                    if gap_end - gap_start <= 1:
                        continue
                    # Try inserting a number in the gap
                    for c in range(gap_start + 1, gap_end):
                        # Check if we can insert c at position i+1
                        new_seq = best_seq_sorted[:i+1] + [c] + best_seq_sorted[i+1:]
                        # Verify strictly increasing
                        if any(new_seq[j] >= new_seq[j+1] for j in range(len(new_seq)-1)):
                            continue
                        # Check all products
                        all_prods = set()
                        valid = True
                        for start_idx in range(len(new_seq)):
                            prod = 1
                            for end_idx in range(start_idx, len(new_seq)):
                                prod *= new_seq[end_idx]
                                if prod in all_prods:
                                    valid = False
                                    break
                                all_prods.add(prod)
                            if not valid:
                                break
                        if valid:
                            best_seq = new_seq
                            best_len = len(best_seq)
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
    
    return best_seq
