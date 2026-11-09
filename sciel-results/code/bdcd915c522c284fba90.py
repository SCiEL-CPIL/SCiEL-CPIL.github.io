"""
Dense increasing sequences with unique block products.

Strategy:
1. Use the greedy algorithm as a baseline (fast, O(n^2) per candidate).
2. Use a backtracking search with pruning to try to extend the sequence beyond greedy.
3. Use local search / hill climbing to try to replace elements to allow more insertions.
4. Time-bounded: return best found within deadline.
"""

import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s margin within 90s limit
    
    # Greedy baseline
    def greedy():
        values = []
        products = set()
        for x in range(2, 129):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values
    
    best = greedy()
    best_len = len(best)
    
    # Verify a sequence
    def is_valid(seq):
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
    
    # Check if a value can be inserted at position pos in seq
    def can_insert(seq, pos, x):
        # Compute all block products in the new sequence
        prods = set()
        n = len(seq)
        # New sequence: seq[0:pos] + [x] + seq[pos:]
        new_seq = seq[:pos] + [x] + seq[pos:]
        m = len(new_seq)
        for i in range(m):
            p = 1
            for j in range(i, m):
                p *= new_seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True
    
    # Try to extend the sequence by inserting values
    def try_extend(seq, candidates, deadline):
        """Try to insert candidate values into seq to make it longer."""
        current = list(seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in candidates:
                if x in current:
                    continue
                for pos in range(len(current) + 1):
                    if time.time() >= deadline:
                        break
                    # Check if x can be inserted at pos
                    # Quick check: x must be greater than current[pos-1] if pos > 0
                    # and less than current[pos] if pos < len(current)
                    if pos > 0 and x <= current[pos-1]:
                        continue
                    if pos < len(current) and x >= current[pos]:
                        continue
                    if can_insert(current, pos, x):
                        current = current[:pos] + [x] + current[pos:]
                        improved = True
                        break
                if time.time() >= deadline:
                    break
        return current
    
    # Try to extend greedy result
    candidates = [x for x in range(2, 129) if x not in best]
    best = try_extend(best, candidates, deadline)
    best_len = len(best)
    
    # Try different greedy orders / heuristics
    # Greedy but skip some values to allow better packing
    def greedy_with_skip(skip_set):
        values = []
        products = set()
        for x in range(2, 129):
            if x in skip_set:
                continue
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values
    
    # Try to find which values to skip
    # First, identify which values are "blocking"
    # A value blocks if removing it allows more values to be added
    
    # Try removing each value from best and see if we can add more
    def try_remove_and_rebuild(base_seq, deadline):
        """Try removing one element from base_seq and rebuilding."""
        base = list(base_seq)
        n = len(base)
        best_result = base
        best_result_len = n
        
        for i in range(n):
            if time.time() >= deadline:
                break
            # Remove element at position i
            modified = base[:i] + base[i+1:]
            # Try to extend
            candidates = [x for x in range(2, 129) if x not in modified]
            extended = try_extend(modified, candidates, deadline)
            if len(extended) > best_result_len:
                best_result = extended
                best_result_len = len(extended)
        
        return best_result
    
    best = try_remove_and_rebuild(best, deadline)
    best_len = len(best)
    
    # Try multiple random restarts with different initial subsets
    import random
    random.seed(42)
    
    def random_greedy(deadline):
        """Build a sequence using a random subset of candidates."""
        values = []
        products = set()
        # Try adding values in a random-ish order but respecting increasing constraint
        # Actually, the sequence must be strictly increasing, so we process in order
        # but can skip some
        for x in range(2, 129):
            if time.time() >= deadline:
                break
            if random.random() < 0.3:  # Skip with some probability
                continue
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values
    
    for _ in range(20):
        if time.time() >= deadline:
            break
        candidate_seq = random_greedy(deadline)
        if len(candidate_seq) > best_len:
            candidates = [x for x in range(2, 129) if x not in candidate_seq]
            candidate_seq = try_extend(candidate_seq, candidates, deadline)
            if len(candidate_seq) > best_len:
                best = candidate_seq
                best_len = len(best)
    
    # Local search: try swapping or replacing elements
    def local_search(seq, deadline):
        current = list(seq)
        current_len = len(current)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() >= deadline:
                    break
                for x in range(2, 129):
                    if x in current:
                        continue
                    # Check if x can replace current[i]
                    # x must maintain increasing order
                    if i > 0 and x <= current[i-1]:
                        continue
                    if i < len(current)-1 and x >= current[i+1]:
                        continue
                    new_seq = current[:i] + [x] + current[i+1:]
                    if is_valid(new_seq):
                        # Try to extend
                        candidates = [y for y in range(2, 129) if y not in new_seq]
                        extended = try_extend(new_seq, candidates, deadline)
                        if len(extended) > current_len:
                            current = extended
                            current_len = len(current)
                            improved = True
                            break
                if time.time() >= deadline:
                    break
        return current
    
    best = local_search(best, deadline)
    best_len = len(best)
    
    return best
