import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    N = 128
    candidates = list(range(2, N + 1))
    
    # Fast validity check
    def is_valid(seq):
        if not seq:
            return True
        if len(set(seq)) != len(seq):
            return False
        if seq != sorted(seq):
            return False
        if any(x < 2 or x > N for x in seq):
            return False
        
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def local_search(seq):
        """
        Improves seq by trying to add or swap elements.
        Returns the best sequence found.
        """
        seq = seq[:]
        current_len = len(seq)
        best_len = current_len
        best_seq = seq[:]
        
        # Sets for fast lookup
        seq_set = set(seq)
        
        # We will iterate for a fixed number of steps or until no improvement
        # Since N is small, we can afford many iterations.
        max_iters = 2000
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try Add
            # Iterate over candidates not in seq
            # Randomize order to escape local minima?
            # For deterministic local search, we can just iterate.
            # But we are calling this from randomized starts, so the start is random.
            # Let's randomize the order of candidates to try.
            
            # 1. Try to add one element
            # We need to check all positions.
            # To speed up, we can just try to append? No, must be sorted.
            # So for each x, find insertion point.
            
            # Optimization: Precompute valid insertions? No, too dynamic.
            
            # Try Add
            add_candidates = [x for x in candidates if x not in seq_set]
            random.shuffle(add_candidates)
            
            added = False
            for x in add_candidates:
                # Find valid positions
                # Binary search for position
                lo, hi = 0, len(seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if seq[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                pos = lo
                
                # Check if inserting at pos is valid
                # We only need to check blocks involving x.
                # Blocks are [i..j] where i <= pos <= j (in new seq)
                # Actually, any block that includes x.
                # x is at index pos in new_seq.
                # Blocks: new_seq[i..j] such that i <= pos <= j.
                # We can check these against all other blocks.
                
                # Efficient check:
                # 1. Compute all products of blocks in current seq. (O(N^2))
                # 2. Compute products of blocks involving x in new seq.
                # 3. Check for duplicates within the new set.
                
                # Since N is small, just construct trial and check is_valid?
                # is_valid is O(N^2).
                # Trial construction is O(N).
                # Total O(N^2) per candidate.
                # 127 candidates * 900 ops = ~100k ops.
                # 2000 iters * 100k = 200M ops. Might be too slow in Python.
                
                # Let's optimize is_valid for the "trial" case.
                # We know seq is valid.
                # We only need to check:
                # 1. x itself is not in existing products.
                # 2. Any block containing x has a product not in existing products.
                # 3. Any two new blocks (both containing x) have distinct products.
                
                # Let's implement a specialized check: `can_insert(seq, x, pos)`
                
                # Existing products
                # We can cache them?
                # For now, let's just use is_valid(trial) but limit iterations if it's slow.
                
                trial = seq[:pos] + [x] + seq[pos:]
                if is_valid(trial):
                    seq = trial
                    seq_set.add(x)
                    current_len += 1
                    if current_len > best_len:
                        best_len = current_len
                        best_seq = seq[:]
                    improved = True
                    added = True
                    break # Restart search with new state
            
            if added:
                continue # Restart the iteration loop to try more adds/swaps
            
            # Try Swap
            # Pick x in seq, y not in seq.
            # Replace x with y.
            # To keep it simple, try swapping each x with each y? O(N^2) pairs.
            # 127*127 = 16k pairs.
            # Check validity for each? 16k * 900 = 14M ops per iteration.
            # Too slow for 2000 iterations.
            
            # Better: Try swapping a random x with a random y.
            if seq:
                x_idx = random.randint(0, len(seq) - 1)
                x = seq[x_idx]
                y = random.choice(add_candidates) # add_candidates is already filtered
                
                # Check if y can replace x
                # Remove x, insert y.
                # Since seq is sorted, y might go to a different position.
                # 1. Remove x.
                # 2. Find pos for y.
                # 3. Insert y.
                
                # Construct trial
                new_seq = seq[:x_idx] + seq[x_idx+1:]
                # Find pos in new_seq
                lo, hi = 0, len(new_seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if new_seq[mid] < y:
                        lo = mid + 1
                    else:
                        hi = mid
                pos = lo
                trial = new_seq[:pos] + [y] + new_seq[pos:]
                
                if is_valid(trial):
                    seq = trial
                    seq_set.remove(x)
                    seq_set.add(y)
                    # Length stays same, but structure changed.
                    # This helps escape local minima where we are stuck with "bad" elements.
                    # We don't update best_len here as length is same.
                    # But it might enable further additions.
                    improved = True
            
            if not improved:
                # No add, no swap succeeded.
                # Try a "kick"? Randomly remove one element?
                # This is dangerous but can help.
                if seq and random.random() < 0.1:
                    rm_idx = random.randint(0, len(seq) - 1)
                    x = seq[rm_idx]
                    seq.pop(rm_idx)
                    seq_set.remove(x)
                    current_len -= 1
                    # Continue to next iteration to try adding something else

        return best_seq

    def greedy_random():
        """
        Randomized greedy construction.
        """
        seq = []
        seq_set = set()
        # Shuffle candidates
        cands = candidates[:]
        random.shuffle(cands)
        
        for x in cands:
            # Try to insert x
            # Find pos
            lo, hi = 0, len(seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if seq[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            pos = lo
            
            trial = seq[:pos] + [x] + seq[pos:]
            if is_valid(trial):
                seq = trial
                seq_set.add(x)
                
        return seq

    best_seq = []
    best_len = 0
    
    # 1. Try the MIS approach from the original code?
    # It's complex to re-include. Let's just rely on Randomized Greedy + Local Search.
    # It's simpler and often better for this kind of constraint satisfaction.
    
    # Run multiple random starts
    num_starts = 50
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Randomized Greedy
        seq = greedy_random()
        
        # Local Search
        improved_seq = local_search(seq)
        
        if len(improved_seq) > best_len:
            best_len = len(improved_seq)
            best_seq = improved_seq
    
    # If best_seq is empty (should not happen), return a valid single element
    if not best_seq:
        best_seq = [2]
        
    return best_seq

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
