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

    def get_all_products(seq):
        """Return set of all block products for a valid sequence."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def can_insert_at(seq, x, pos, existing_prods):
        """
        Check if inserting x at position pos in seq keeps all block products distinct.
        existing_prods is the set of all block products of seq.
        Returns True if valid, False otherwise.
        """
        # Compute products of all blocks that include x in the new sequence
        new_prods = set()
        n = len(seq)
        
        # Blocks that include x: [i..j] in new_seq where i <= pos <= j
        # In terms of original seq indices: i_orig in [0..pos-1] or i_orig = pos (x itself)
        # Actually, let's think in terms of new_seq indices.
        # new_seq = seq[:pos] + [x] + seq[pos:]
        # Block [i..j] in new_seq includes x iff i <= pos <= j.
        # The product is (product of seq[i..pos-1]) * x * (product of seq[pos..j-1])
        # Wait, need to be careful with indexing.
        
        # Let's just compute all blocks containing x and check against existing_prods and each other.
        # Also check x itself.
        
        # Prefix products from left: prod_left[k] = product of seq[k..pos-1] for k in [0..pos]
        # Actually, simpler: iterate over all i, j such that block [i..j] in new_seq contains x.
        # i can range from 0 to pos, j can range from pos to n (in new_seq, which has n+1 elements)
        # In new_seq, x is at index pos.
        # Block [i..j] contains x iff i <= pos <= j.
        # i in [0, pos], j in [pos, n] (since new_seq has indices 0..n)
        
        # But we need to compute the product efficiently.
        # Let's precompute prefix and suffix products around x.
        
        # Left part: seq[0..pos-1]
        # Right part: seq[pos..n-1]
        
        # For a block [i..j] in new_seq containing x:
        # If i <= pos and j >= pos:
        #   left_part = product of new_seq[i..pos-1] = product of seq[i..pos-1] (if i < pos)
        #   right_part = product of new_seq[pos+1..j] = product of seq[pos..j-1] (if j > pos)
        #   product = left_part * x * right_part
        
        # Let's compute:
        # left_prods[i] for i in [0..pos]: product of seq[i..pos-1]
        #   left_prods[pos] = 1 (empty)
        #   left_prods[i] = seq[i] * left_prods[i+1] for i from pos-1 down to 0
        
        # right_prods[j] for j in [pos..n]: product of seq[pos..j-1]
        #   right_prods[pos] = 1 (empty)
        #   right_prods[j] = right_prods[j-1] * seq[j-1] for j from pos+1 to n
        
        left_prods = [1] * (pos + 1)
        for i in range(pos - 1, -1, -1):
            left_prods[i] = left_prods[i + 1] * seq[i]
        
        right_prods = [1] * (n - pos + 1)
        for j in range(pos + 1, n + 1):
            right_prods[j] = right_prods[j - 1] * seq[j - 1]
        
        # Now check all blocks [i..j] in new_seq where i <= pos <= j
        # i in [0, pos], j in [pos, n]
        for i in range(pos + 1):
            for j in range(pos, n + 1):
                # Block [i..j] in new_seq
                # left_part = left_prods[i] (product of seq[i..pos-1])
                # right_part = right_prods[j] (product of seq[pos..j-1])
                # Wait, right_prods[j] is product of seq[pos..j-1], which is new_seq[pos+1..j]
                # So product = left_prods[i] * x * right_prods[j]
                prod = left_prods[i] * x * right_prods[j]
                
                if prod in existing_prods:
                    return False
                if prod in new_prods:
                    return False
                new_prods.add(prod)
        
        return True

    def local_search(seq, time_budget=5.0):
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
        
        # Precompute existing products
        existing_prods = get_all_products(seq)
        
        start = time.time()
        max_iters = 5000
        
        for _ in range(max_iters):
            if time.time() - start > time_budget or time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try Add
            add_candidates = [x for x in candidates if x not in seq_set]
            random.shuffle(add_candidates)
            
            added = False
            for x in add_candidates:
                # Find insertion position
                lo, hi = 0, len(seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if seq[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                pos = lo
                
                # Check if inserting at pos is valid
                if can_insert_at(seq, x, pos, existing_prods):
                    # Update seq
                    seq = seq[:pos] + [x] + seq[pos:]
                    seq_set.add(x)
                    current_len += 1
                    
                    # Update existing_prods incrementally
                    # New products are all blocks containing x
                    n = len(seq)
                    pos_new = pos  # x is at pos in new seq
                    # Recompute is expensive; let's just recompute for simplicity if needed
                    # But for now, let's recompute
                    existing_prods = get_all_products(seq)
                    
                    if current_len > best_len:
                        best_len = current_len
                        best_seq = seq[:]
                    improved = True
                    added = True
                    break  # Restart search with new state
            
            if added:
                continue  # Restart the iteration loop to try more adds/swaps
            
            # Try Swap: replace x in seq with y not in seq
            if seq and add_candidates:
                # Try a few random swaps
                for _swap_attempt in range(10):
                    x_idx = random.randint(0, len(seq) - 1)
                    x = seq[x_idx]
                    y = random.choice(add_candidates)
                    
                    # Construct trial: remove x, insert y
                    new_seq = seq[:x_idx] + seq[x_idx+1:]
                    # Find pos for y in new_seq
                    lo, hi = 0, len(new_seq)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if new_seq[mid] < y:
                            lo = mid + 1
                        else:
                            hi = mid
                    pos_y = lo
                    
                    # Check validity
                    trial = new_seq[:pos_y] + [y] + new_seq[pos_y:]
                    if is_valid(trial):
                        seq = trial
                        seq_set.remove(x)
                        seq_set.add(y)
                        existing_prods = get_all_products(seq)
                        improved = True
                        break
            
            if not improved:
                # No add, no swap succeeded.
                # Try a "kick"? Randomly remove one element?
                if seq and random.random() < 0.15:
                    rm_idx = random.randint(0, len(seq) - 1)
                    x = seq[rm_idx]
                    seq.pop(rm_idx)
                    seq_set.remove(x)
                    current_len -= 1
                    existing_prods = get_all_products(seq)
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

    def greedy_sorted():
        """
        Greedy construction trying candidates in sorted order.
        """
        seq = []
        for x in candidates:
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
        return seq

    def greedy_descending():
        """
        Greedy construction trying candidates in descending order.
        """
        seq = []
        for x in reversed(candidates):
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
        return seq

    best_seq = []
    best_len = 0
    
    # Strategy: Try multiple approaches and keep the best
    # 1. Sorted greedy
    if time.time() < deadline - 20:
        seq = greedy_sorted()
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
    
    # 2. Descending greedy
    if time.time() < deadline - 20:
        seq = greedy_descending()
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
    
    # 3. Multiple randomized greedy + local search
    num_starts = 30
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Randomized Greedy
        seq = greedy_random()
        
        # Local Search with time budget
        remaining = deadline - 10 - time.time()
        time_budget = min(5.0, remaining / max(1, num_starts - i))
        improved_seq = local_search(seq, time_budget=time_budget)
        
        if len(improved_seq) > best_len:
            best_len = len(improved_seq)
            best_seq = improved_seq
    
    # 4. Local search from best found so far with more time
    if best_seq and time.time() < deadline - 8:
        remaining = deadline - 8 - time.time()
        improved_seq = local_search(best_seq, time_budget=remaining)
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
