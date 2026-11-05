import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_instance(values, time_limit):
        n = len(values)
        val_to_idx = {v: i for i, v in enumerate(values)}
        
        # Find all triples (a, b, c) such that a + b = c
        # a, b, c are indices
        triples = []
        for i in range(n):
            for j in range(n):
                s = values[i] + values[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Store unique triples? 
                    # (i, j, k) and (j, i, k) are the same constraint for the set.
                    # We just need to know if i, j, k are all in the set.
                    triples.append((i, j, k))
        
        # Initial solution: Large absolute values
        # Sort by abs value
        sorted_by_abs = sorted(range(n), key=lambda i: -abs(values[i]))
        mid_idx = n // 2
        # Take the top half by absolute value
        # This is sum-free because if |a|, |b| > M/2, then |a+b| > M/2 is not guaranteed?
        # Wait. If a, b are in (M/2, M], a+b > M. Not in set.
        # If a, b are in [-M, -M/2), a+b < -M. Not in set.
        # If a in (M/2, M], b in [-M, -M/2), a+b in (-M/2, M/2).
        # If the set ONLY contains elements with |x| > M/2, then a+b is in (-M/2, M/2) or outside [-M, M].
        # Since the set contains NO elements with |x| <= M/2, a+b cannot be in the set.
        # So this is a valid sum-free set.
        
        init_set = set(sorted_by_abs[:mid_idx])
        
        # Check if we can do better with odds?
        odd_set = set(i for i in range(n) if values[i] % 2 != 0)
        if len(odd_set) > len(init_set):
            init_set = odd_set
            
        best_set = set(init_set)
        best_size = len(best_set)
        
        # Simulated Annealing
        # State: set of indices
        current_set = set(best_set)
        current_size = len(current_set)
        
        # Pre-compute conflicts for each element?
        # conflict_count[i] = number of triples containing i that are currently "active" (all 3 in set)
        # Actually, we just need to check validity of the whole set.
        
        def is_sum_free(s):
            # Check all triples
            for i, j, k in triples:
                if i in s and j in s and k in s:
                    return False
            return True
        
        # Verify initial
        if not is_sum_free(current_set):
             # Fallback to empty or simple
             current_set = set()
             current_size = 0

        T = 10.0
        min_T = 0.01
        cooling = 0.999
        
        steps = 0
        while time.time() < time_limit:
            # Pick a random move
            # 1. Add a random element not in set
            # 2. Remove a random element in set
            # 3. Swap
            
            if random.random() < 0.5:
                # Add
                candidates = [i for i in range(n) if i not in current_set]
                if not candidates:
                    break
                i = random.choice(candidates)
                new_set = current_set | {i}
            else:
                # Remove
                if not current_set:
                    continue
                i = random.choice(list(current_set))
                new_set = current_set - {i}
            
            new_size = len(new_set)
            delta_size = new_size - current_size
            
            # Check validity
            # Instead of checking all triples, we can check only those involving i
            # But for n=240, checking all is fast enough?
            # Number of triples is ~n^2 = 57600.
            # 57600 checks per step.
            # 120s / 57600 * (ops per check) 
            # Python is slow. 
            # Let's optimize the check.
            
            # Pre-compute for each index i, the list of triples it participates in.
            # triples_by_idx[i] = list of (j, k) such that (i, j, k) is a triple (in any position)
            # Actually, a triple (a,b,c) is violated if a,b,c are all in S.
            # If we add i, we only need to check triples containing i.
            # If we remove i, we only need to check triples containing i (they become valid).
            
            # Let's build this index.
            # This should be done outside the loop.
            
            valid = True
            # Check triples involving i
            for t in triples_by_idx[i]:
                a, b, c = t
                if a in new_set and b in new_set and c in new_set:
                    valid = False
                    break
            
            if valid:
                if delta_size > 0:
                    # Always accept if size increases
                    current_set = new_set
                    current_size = new_size
                    if current_size > best_size:
                        best_set = set(current_set)
                        best_size = current_size
                else:
                    # Accept with probability exp(delta_size / T) ? 
                    # delta_size is negative or 0.
                    # We want to maximize size, so we penalize size decrease.
                    # Accept if random < exp(delta_size / T)
                    if random.random() < math.exp(delta_size / T):
                        current_set = new_set
                        current_size = new_size
            else:
                # Invalid state. 
                # Penalty for invalidity?
                # We can just reject.
                # Or accept with very low prob to escape local minima?
                # Since we want a valid set, rejecting is safer.
                pass
            
            T *= cooling
            if T < min_T:
                T = min_T
            steps += 1
            
        # Final polish: Try to add any remaining element
        for i in range(n):
            if i not in best_set:
                test_set = best_set | {i}
                valid = True
                for t in triples_by_idx[i]:
                    a, b, c = t
                    if a in test_set and b in test_set and c in test_set:
                        valid = False
                        break
                if valid:
                    best_set = test_set
                    best_size += 1
                    
        return [values[i] for i in best_set]

    # Build triples_by_idx for each instance
    results = []
    time_per_instance = (deadline - time.time()) / 3.0
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        n = len(vals)
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        triples = []
        triples_by_idx = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                s = vals[i] + vals[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Triple (i, j, k)
                    triples.append((i, j, k))
                    triples_by_idx[i].append((i, j, k))
                    triples_by_idx[j].append((i, j, k))
                    triples_by_idx[k].append((i, j, k))
        
        # Pass triples_by_idx to solver
        # We need to modify solve_instance to accept triples_by_idx
        # Or just inline it.
        
        # Let's refactor solve_instance to take pre-computed data
        # To avoid code duplication, I'll just call a helper.
        
        # For now, I'll just re-implement the core loop here to use the pre-computed data.
        
        init_set = set()
        sorted_by_abs = sorted(range(n), key=lambda i: -abs(vals[i]))
        mid_idx = n // 2
        init_set = set(sorted_by_abs[:mid_idx])
        
        odd_set = set(i for i in range(n) if vals[i] % 2 != 0)
        if len(odd_set) > len(init_set):
            init_set = odd_set
            
        best_set = set(init_set)
        best_size = len(best_set)
        
        current_set = set(best_set)
        current_size = len(current_set)
        
        T = 10.0
        min_T = 0.01
        cooling = 0.999
        
        while time.time() < deadline - 2.0: # Leave 2s for last instance
            if random.random() < 0.5:
                candidates = [i for i in range(n) if i not in current_set]
                if not candidates:
                    continue
                i = random.choice(candidates)
                new_set = current_set | {i}
            else:
                if not current_set:
                    continue
                i = random.choice(list(current_set))
                new_set = current_set - {i}
            
            new_size = len(new_set)
            delta_size = new_size - current_size
            
            valid = True
            for t in triples_by_idx[i]:
                a, b, c = t
                if a in new_set and b in new_set and c in new_set:
                    valid = False
                    break
            
            if valid:
                if delta_size > 0:
                    current_set = new_set
                    current_size = new_size
                    if current_size > best_size:
                        best_set = set(current_set)
                        best_size = current_size
                else:
                    if random.random() < math.exp(delta_size / T):
                        current_set = new_set
                        current_size = new_size
            else:
                pass
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        # Final polish
        for i in range(n):
            if i not in best_set:
                test_set = best_set | {i}
                valid = True
                for t in triples_by_idx[i]:
                    a, b, c = t
                    if a in test_set and b in test_set and c in test_set:
                        valid = False
                        break
                if valid:
                    best_set = test_set
                    best_size += 1
        
        results.append([vals[i] for i in best_set])
        
    return results

# We need to define triples_by_idx inside the loop or pass it.
# The code above has a flaw: triples_by_idx is defined in the loop but used in a hypothetical solve_instance.
# I will write the clean version below.
