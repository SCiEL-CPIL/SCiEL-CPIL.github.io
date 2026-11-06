import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def get_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def is_valid(seq):
        prods = get_products(seq)
        n = len(seq)
        num_blocks = n * (n + 1) // 2
        return len(prods) == num_blocks

    def try_insert(seq, x, products):
        """Try to insert x into sorted seq at its correct position.
        Returns (new_seq, new_products) if valid, else (None, None)."""
        if x in seq:
            return None, None
        
        # Find insertion index
        lo, hi = 0, len(seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        k = lo
        
        n = len(seq)
        
        # Compute left_prods[i] = product of seq[i..k-1] for i in 0..k
        # left_prods[k] = 1 (empty left part)
        left_prods = [1] * (k + 1)
        for i in range(k - 1, -1, -1):
            left_prods[i] = left_prods[i + 1] * seq[i]
        
        # Compute right_prods[j] = product of seq[k..j] for j in k..n-1
        # We also need the empty right part = 1
        # Let's use a list right_prods where right_prods[t] corresponds to
        # the product from seq[k] to seq[k+t-1] for t >= 1, and right_prods[0] = 1
        right_prods = [1] * (n - k + 1)
        for t in range(1, n - k + 1):
            right_prods[t] = right_prods[t - 1] * seq[k + t - 1]
        
        # Check all combinations of new blocks containing x
        # Block is defined by left index i (0..k) and right length t (0..n-k)
        # Product = left_prods[i] * x * right_prods[t]
        new_products = set(products)
        seen_new = set()
        
        for i in range(k + 1):
            lp = left_prods[i]
            for t in range(n - k + 1):
                p = lp * x * right_prods[t]
                if p in new_products:
                    return None, None
                if p in seen_new:
                    return None, None
                seen_new.add(p)
        
        new_products.update(seen_new)
        new_seq = seq[:k] + [x] + seq[k:]
        return new_seq, new_products

    def greedy_sorted():
        values, products = [], set()
        for x in range(2, 129):
            new_seq, new_prods = try_insert(values, x, products)
            if new_seq is not None:
                values = new_seq
                products = new_prods
        return values

    def greedy_random():
        values, products = [], set()
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        for x in candidates:
            new_seq, new_prods = try_insert(values, x, products)
            if new_seq is not None:
                values = new_seq
                products = new_prods
        return values

    def polish(seq, products):
        """Try to insert any missing number at its valid position."""
        changed = True
        while changed:
            changed = False
            for x in range(2, 129):
                if x in seq:
                    continue
                new_seq, new_prods = try_insert(seq, x, products)
                if new_seq is not None:
                    seq = new_seq
                    products = new_prods
                    changed = True
                    break
        return seq, products

    def backtrack_search(seq, products, max_depth=3):
        """Try to remove one element and add two to increase length."""
        n = len(seq)
        if n == 0:
            return seq, products
        
        # Try removing each element
        for i in range(n):
            # Remove seq[i]
            new_seq = seq[:i] + seq[i+1:]
            
            # Recompute products
            new_prods = get_products(new_seq)
            
            # Try adding two missing numbers
            missing = [x for x in range(2, 129) if x not in new_seq]
            
            for x in missing[:10]:  # Limit to first 10 missing for speed
                s1, p1 = try_insert(new_seq, x, new_prods)
                if s1 is None:
                    continue
                for y in missing:
                    if y == x:
                        continue
                    s2, p2 = try_insert(s1, y, p1)
                    if s2 is not None and len(s2) > n:
                        return s2, p2
        
        return seq, products

    # Initial best from deterministic greedy
    best_seq = greedy_sorted()
    best_prods = get_products(best_seq)
    best_len = len(best_seq)
    
    # Polish the best
    best_seq, best_prods = polish(best_seq, best_prods)
    best_len = len(best_seq)
    
    # Randomized search
    while time.time() < deadline:
        # Randomized Greedy Construction
        current = greedy_random()
        
        # Polish
        current_prods = get_products(current)
        current, current_prods = polish(current, current_prods)
        
        # Backtracking if we're close
        if len(current) < best_len + 2:
            current, current_prods = backtrack_search(current, current_prods)
        
        if len(current) > best_len:
            best_len = len(current)
            best_seq = current
            best_prods = current_prods

    return best_seq

if __name__ == "__main__":
    seq = construct()
    print(seq)
    print(len(seq))
