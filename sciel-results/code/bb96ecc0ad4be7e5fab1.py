"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations

def construct():
    deadline = time.time() + 80  # 80s safety margin (limit is 90s)
    
    def compute_sums(seq):
        """Compute all consecutive block sums for a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums
    
    def suffix_sums(seq):
        """Compute all suffix sums (sums of consecutive blocks ending at the last element)."""
        result = []
        total = 0
        for a in reversed(seq):
            total += a
            result.append(total)
        return result
    
    def beam_search(max_beam=200, max_depth=20):
        """Beam search maintaining multiple partial sequences."""
        if time.time() > deadline:
            return []
        
        # Each state: (negative_length, seq, sums_set, last_val)
        # We want to maximize length, so use negative for min-heap
        beam = [([], set(), 0)]  # Start with empty sequence
        best_seq = []
        
        for depth in range(max_depth):
            if time.time() > deadline:
                break
            
            candidates = []
            for seq, sums, last in beam:
                if time.time() > deadline:
                    break
                
                # Try adding each possible next value
                start = last + 1
                # Pruning: if we can't beat best, skip
                if len(seq) + (256 - start + 1) <= len(best_seq):
                    continue
                
                for x in range(start, 257):
                    # Check if x creates a conflict
                    new_suffixes = [x]
                    total = x
                    for a in reversed(seq):
                        total += a
                        new_suffixes.append(total)
                    
                    # Check against existing sums
                    conflict = False
                    for s in new_suffixes:
                        if s in sums:
                            conflict = True
                            break
                    
                    if conflict:
                        continue
                    
                    # Also check: new suffixes among themselves are distinct (they are, since increasing)
                    # And new suffixes don't conflict with each other (they can't since strictly increasing seq)
                    # But we need to check that new_suffixes don't have duplicates - they won't for increasing seq
                    
                    new_sums = sums | set(new_suffixes)
                    candidates.append((-(len(seq) + 1), list(seq + [x]), new_sums, x))
            
            if not candidates:
                break
            
            # Sort by length (descending), then by last value (ascending) for lexicographic preference
            candidates.sort(key=lambda c: (c[0], c[3]))
            
            # Keep top max_beam candidates
            new_beam = []
            seen = set()
            for neg_len, seq, sums, last in candidates:
                # Use a hash of the sequence to avoid duplicates
                key = tuple(seq)
                if key in seen:
                    continue
                seen.add(key)
                new_beam.append((seq, sums, last))
                
                if len(seq) > len(best_seq):
                    best_seq = list(seq)
                
                if len(new_beam) >= max_beam:
                    break
            
            if not new_beam:
                break
            
            beam = new_beam
        
        return best_seq
    
    def iterative_deepening_dfs(max_len, seq, sums, last, best_holder):
        """DFS with iterative deepening to find sequences of exactly max_len."""
        if time.time() > deadline:
            return False
        
        if len(seq) == max_len:
            if len(seq) > len(best_holder[0]):
                best_holder[0] = list(seq)
            return True
        
        # Pruning: not enough values left
        if len(seq) + (256 - last) < max_len:
            return False
        
        found = False
        for x in range(last + 1, 257):
            if time.time() > deadline:
                return False
            
            # Compute new suffixes
            new_suffixes = [x]
            total = x
            for a in reversed(seq):
                total += a
                new_suffixes.append(total)
            
            # Check conflicts
            conflict = False
            for s in new_suffixes:
                if s in sums:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            new_sums = sums | set(new_suffixes)
            seq.append(x)
            
            if iterative_deepening_dfs(max_len, seq, new_sums, x, best_holder):
                found = True
                seq.pop()
                return True
            
            seq.pop()
        
        return found
    
    def hybrid_search():
        """Combine beam search with targeted DFS extensions."""
        if time.time() > deadline:
            return []
        
        best_seq = []
        
        # Phase 1: Beam search to get a good starting point
        beam_result = beam_search(max_beam=300, max_depth=25)
        if beam_result:
            best_seq = beam_result
        
        if time.time() > deadline:
            return best_seq
        
        # Phase 2: Try to extend the best sequence found by beam search
        if best_seq:
            best_sums = compute_sums(best_seq)
            best_holder = [best_seq]
            
            # Try DFS extension from the beam result
            current_len = len(best_seq)
            for target_len in range(current_len + 1, min(current_len + 15, 257)):
                if time.time() > deadline:
                    break
                
                # Reset and try to find a sequence of target_len
                if iterative_deepening_dfs(target_len, list(best_seq), best_sums, best_seq[-1], best_holder):
                    best_seq = best_holder[0]
                    best_sums = compute_sums(best_seq)
                else:
                    break
        
        # Phase 3: Multiple beam searches with different parameters
        if time.time() < deadline - 10:
            for beam_size in [100, 500, 1000]:
                if time.time() > deadline:
                    break
                
                result = beam_search(max_beam=beam_size, max_depth=30)
                if result and len(result) > len(best_seq):
                    best_seq = result
        
        # Phase 4: Try to extend the overall best
        if best_seq and time.time() < deadline - 5:
            best_sums = compute_sums(best_seq)
            best_holder = [best_seq]
            current_len = len(best_seq)
            
            for target_len in range(current_len + 1, min(current_len + 10, 257)):
                if time.time() > deadline:
                    break
                
                if iterative_deepening_dfs(target_len, list(best_seq), best_sums, best_seq[-1], best_holder):
                    best_seq = best_holder[0]
                    best_sums = compute_sums(best_seq)
                else:
                    break
        
        return best_seq
    
    result = hybrid_search()
    
    # Final validation
    if result:
        s = compute_sums(result)
        if len(s) != len(result) * (len(result) + 1) // 2:
            # Invalid - fall back to a simple greedy
            values = []
            sums = set()
            for x in range(1, 257):
                tails = [x]
                total = x
                for a in reversed(values):
                    total += a
                    tails.append(total)
                if not sums.intersection(tails):
                    values.append(x)
                    sums.update(tails)
            result = values
    
    return result
# EVOLVE-BLOCK-END
