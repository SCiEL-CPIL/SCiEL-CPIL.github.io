import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    def compute_products(seq):
        """Compute all consecutive block products for a sequence."""
        n = len(seq)
        if n == 0:
            return []
        prods = []
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        return prods
    
    def count_conflicts(seq):
        """Count number of duplicate product pairs."""
        if len(seq) < 2:
            return 0
        prods = compute_products(seq)
        c = Counter(prods)
        conflicts = 0
        for v in c.values():
            if v > 1:
                conflicts += v * (v - 1) // 2
        return conflicts

    def get_score(seq):
        """Score: length if valid, else length - penalty * conflicts."""
        if not seq:
            return 0
        conf = count_conflicts(seq)
        if conf == 0:
            return len(seq)
        return len(seq) - 100 * conf # Increase penalty to strongly discourage invalid states

    def greedy_construct(randomize=False, skip_prob=0.0):
        """Greedy: include x if it doesn't create duplicate products."""
        values = []
        products = set()
        for x in range(2, 129):
            if randomize and random.random() < skip_prob:
                continue
            # New products when adding x: x, x*last, x*last*second_last, ...
            new_tails = [x]
            p = x
            for a in reversed(values):
                p *= a
                new_tails.append(p)
            if not products.intersection(new_tails):
                values.append(x)
                products.update(new_tails)
        return values
    
    # Multi-start initialization
    best_seq = []
    best_len = 0
    
    # Run several randomized greedy passes to find a good start
    for i in range(20):
        seq = greedy_construct(randomize=True, skip_prob=0.05)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
    # Also try the deterministic greedy
    seq_det = greedy_construct(randomize=False)
    if len(seq_det) > best_len:
        best_len = len(seq_det)
        best_seq = seq_det[:]

    # Simulated Annealing
    current_seq = best_seq[:]
    current_score = get_score(current_seq)
    
    T = 10.0
    cool_rate = 0.999
    best_T = T
    no_improve = 0
    
    # Precompute set of all possible values for faster random choice?
    all_vals = list(range(2, 129))
    
    while time.time() < deadline - 3:
        # Generate a neighbor
        r = random.random()
        
        if r < 0.4:
            # Try to add an element
            in_seq = set(current_seq)
            available = [x for x in all_vals if x not in in_seq]
            if not available:
                continue
            x = random.choice(available)
            # Insert in sorted position
            pos = 0
            for i, v in enumerate(current_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            new_seq = current_seq[:pos] + [x] + current_seq[pos:]
        elif r < 0.8:
            # Try to remove an element
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            new_seq = current_seq[:idx] + current_seq[idx+1:]
        else:
            # Try a Swap: Remove one, Add one
            if not current_seq:
                continue
            in_seq = set(current_seq)
            available = [x for x in all_vals if x not in in_seq]
            if not available:
                continue
            
            # Pick element to remove
            idx_remove = random.randint(0, len(current_seq) - 1)
            x_remove = current_seq[idx_remove]
            
            # Pick element to add
            x_add = random.choice(available)
            
            # Construct new sequence: remove x_remove, add x_add
            # Since we need to keep it sorted, we can just make a set, sort, and check
            # But list operations are faster for small n.
            new_seq_set = set(current_seq)
            new_seq_set.remove(x_remove)
            new_seq_set.add(x_add)
            new_seq = sorted(new_seq_set)
        
        new_score = get_score(new_seq)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/T) for worse
        if delta >= 0 or random.random() < np.exp(delta / T):
            current_seq = new_seq
            current_score = new_score
            
            # Check if this is a new best valid solution
            if count_conflicts(current_seq) == 0:
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    no_improve = 0
                    # Reset temperature occasionally when finding new best
                    T = max(T, 5.0)
            else:
                no_improve += 1
        
        # Cooling
        T *= cool_rate
        if T < 0.01:
            T = 0.01
        
        # Restart if stuck
        if no_improve > 1000:
            no_improve = 0
            T = best_T * 0.5
            # Perturb current solution: randomly remove a few elements
            if current_seq:
                for _ in range(min(3, len(current_seq))):
                    idx = random.randint(0, len(current_seq) - 1)
                    current_seq = current_seq[:idx] + current_seq[idx+1:]
                current_score = get_score(current_seq)
    
    # Final polish: try to add any missing elements that don't create conflicts
    in_best = set(best_seq)
    for x in range(2, 129):
        if x not in in_best:
            pos = 0
            for i, v in enumerate(best_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            test_seq = best_seq[:pos] + [x] + best_seq[pos:]
            if count_conflicts(test_seq) == 0:
                best_seq = test_seq
                in_best.add(x)
    
    return best_seq
