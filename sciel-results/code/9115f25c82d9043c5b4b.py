import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
    # 1. Run Greedy to get a good initial lower bound
    def run_greedy():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values, products
    
    best_seq, best_prods = run_greedy()
    best_len = len(best_seq)
    
    # 2. Beam Search
    # State: seq_tuple -> prods_set
    # We start with the empty state
    beam = {(): set()}
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        next_beam = {}
        
        for seq, prods in beam.items():
            # Option 1: Skip x
            # The state (seq, prods) is already in next_beam? 
            # We need to merge states. If we skip x, the state remains (seq, prods).
            # If multiple paths lead to the same seq, we keep the one with the "best" prods?
            # But prods is determined by seq. So if seq is same, prods is same.
            # So we just need to ensure the state is in next_beam.
            if seq not in next_beam:
                next_beam[seq] = prods
            else:
                # This shouldn't happen if we deduplicate by seq, 
                # but if we did, prods would be identical.
                pass
            
            # Option 2: Include x
            # Check if x can be appended
            # seq is strictly increasing, so if seq is empty or seq[-1] < x, it's valid structurally.
            # Since we iterate x in increasing order and build seqs by appending x, 
            # any seq in the beam at step x-1 has elements < x.
            
            # Compute tails
            tails = [x]
            prod = x
            for a in reversed(seq):
                prod *= a
                tails.append(prod)
                
            # Check collision
            # Optimization: check if any tail is in prods
            # If prods is large, this is O(len(tails)).
            has_collision = False
            for t in tails:
                if t in prods:
                    has_collision = True
                    break
            
            if not has_collision:
                new_seq = seq + (x,)
                # new_prods = prods.union(tails)
                # We can lazily create the new set? 
                # We need it for future collision checks.
                new_prods = set(prods)
                new_prods.update(tails)
                
                # Add to next_beam
                # If new_seq is already in next_beam (from a skip path? No, skip keeps old seq. 
                # From another include path? No, x is fixed. 
                # So new_seq is unique for this x step unless we had duplicates in beam).
                # We deduplicate beam by seq, so new_seq is new.
                next_beam[new_seq] = new_prods
        
        # Prune beam
        # Sort by length desc, then by sum of elements asc (smaller sum is better)
        sorted_seqs = sorted(next_beam.keys(), key=lambda s: (-len(s), sum(s) if s else 0))
        
        # Keep top W
        W = 300
        beam = {s: next_beam[s] for s in sorted_seqs[:W]}
        
        # Update best
        if len(sorted_seqs[0]) > best_len:
            best_len = len(sorted_seqs[0])
            best_seq = list(sorted_seqs[0])
            best_prods = next_beam[sorted_seqs[0]]
            
    return list(best_seq)
