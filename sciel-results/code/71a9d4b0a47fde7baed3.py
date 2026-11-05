"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 142  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_one(vals, time_budget=35.0):
        n = len(vals)
        if n == 0:
            return []
        
        vals_set = set(vals)
        index_of = {v: i for i, v in enumerate(vals)}
        
        # Precompute conflict masks
        # For each element i, precompute:
        # 1. sum_mask[i]: bitmask of j such that vals[i] + vals[j] is in vals_set
        # 2. double_mask[i]: bitmask of k such that 2*vals[i] = vals[k] (i.e., i is "half" of k)
        # Actually, we need:
        # can_add(S, x): for all a in S, (a+x) not in S AND 2*x not in S
        
        # Let's precompute `partner_mask[i]`: bitmask of j such that vals[i] + vals[j] is in vals_set.
        # This is symmetric.
        # To check if we can add x (index xi) to S:
        # We need: (S & partner_mask[xi]) == 0 AND (2*vals[xi] not in S)
        # Wait, if vals[i] + vals[j] = vals[k], and we add i, we must ensure that for no j in S is k in S.
        # So we need: for all j in S, (vals[i] + vals[j]) is NOT in S.
        # Let `sum_target[i]` be the set of indices k such that vals[i] + vals[k] = vals[m] for some m? No.
        # Let's use a different precomputation.
        # For each value v in vals, let `sums_with_v` be the set of values w in vals such that v+w is in vals.
        # This is not quite right either.
        
        # Let's just use the O(n) check with a set for now, it's fast enough for n<=240.
        # can_add(S_set, x):
        #   if 2*x in S_set: return False
        #   for a in S_set:
        #       if a+x in S_set: return False
        #   return True
        
        def can_add(S_set, x):
            if 2 * x in S_set:
                return False
            for a in S_set:
                if a + x in S_set:
                    return False
            return True

        # Precompute all conflict triples (i, j, k) where vals[i] + vals[j] = vals[k]
        # This helps in WalkSAT-style clause selection.
        conflict_triples = []
        for i in range(n):
            for j in range(n):
                s = vals[i] + vals[j]
                if s in vals_set:
                    k = index_of[s]
                    conflict_triples.append((i, j, k))
        
        # For faster lookup: given a set S, which triples are violated?
        # We can maintain a count of violated clauses per variable.
        
        # Strategy: Simulated Annealing / Randomized Local Search
        # State: a set S (or bitmask)
        # Objective: maximize |S| subject to sum-free constraint.
        # We'll use a "repair" approach:
        # 1. Start with a large random set or greedy set.
        # 2. If there are violations, remove elements to fix them.
        # 3. Try to add elements back.
        
        # Let's use a bitmask representation for speed.
        # n <= 240, so we need a big int or a list of bools.
        # Python's int is arbitrary precision, so we can use it as a bitmask.
        
        all_mask = (1 << n) - 1
        
        # Precompute for each i, the mask of j such that vals[i] + vals[j] is in vals_set
        # And for each i, the mask of k such that 2*vals[i] = vals[k]
        
        # Actually, let's precompute `sum_mask[i]`: bitmask of j such that vals[i] + vals[j] is in vals_set.
        # And `double_mask[i]`: bitmask of k such that 2*vals[i] = vals[k].
        
        sum_mask = [0] * n
        double_mask = [0] * n
        
        for i in range(n):
            for j in range(n):
                s = vals[i] + vals[j]
                if s in vals_set:
                    k = index_of[s]
                    sum_mask[i] |= (1 << j)
            
            d = 2 * vals[i]
            if d in vals_set:
                k = index_of[d]
                double_mask[i] |= (1 << k)
        
        # To check if we can add element i to set S (bitmask):
        # 1. 2*vals[i] is not in S: (S & double_mask[i]) == 0
        # 2. For all j in S, vals[i] + vals[j] is not in S.
        #    This is equivalent to: (S & sum_mask[i]) has no j such that vals[i] + vals[j] is in S.
        #    Wait, sum_mask[i] contains j such that vals[i] + vals[j] is in vals_set.
        #    But we need vals[i] + vals[j] to be IN S.
        #    So this check is not directly expressible as a single bitmask AND.
        #    We need: for all j in S, (vals[i] + vals[j]) not in S.
        #    Let T = S. We need T & (T + vals[i]) == 0? Not quite.
        
        # Let's stick to the O(n) check for can_add, it's simple and n is small.
        
        def can_add_bitmask(S_mask, i):
            # Check 2*vals[i] not in S
            if S_mask & double_mask[i]:
                return False
            # Check vals[i] + vals[j] not in S for all j in S
            x = vals[i]
            # Iterate over set bits in S_mask
            m = S_mask
            while m:
                # Get lowest set bit
                lb = m & (-m)
                j = lb.bit_length() - 1
                if x + vals[j] in vals_set:
                    k = index_of[x + vals[j]]
                    if S_mask & (1 << k):
                        return False
                m ^= lb
            return True

        # Greedy initialization with multiple random orderings
        best_mask = 0
        best_size = 0
        
        rng = random.Random(12345)
        
        # Try 10 greedy inits
        for _ in range(10):
            S_mask = 0
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                if can_add_bitmask(S_mask, i):
                    S_mask |= (1 << i)
            size = bin(S_mask).count('1')
            if size > best_size:
                best_size = size
                best_mask = S_mask
        
        # Also try "odd" initialization
        odd_mask = 0
        for i, v in enumerate(vals):
            if v % 2 != 0:
                odd_mask |= (1 << i)
        if bin(odd_mask).count('1') > best_size:
            best_size = bin(odd_mask).count('1')
            best_mask = odd_mask
        
        # Local search / Simulated Annealing
        # We'll use a "kick" strategy: randomly remove some elements, then repair.
        
        end_time = start_time + time_budget
        current_mask = best_mask
        current_size = best_size
        
        while time.time() < end_time:
            # Try to add elements greedily
            improved = True
            while improved:
                improved = False
                # Try each element not in current_mask
                for i in range(n):
                    if not (current_mask & (1 << i)):
                        if can_add_bitmask(current_mask, i):
                            current_mask |= (1 << i)
                            current_size += 1
                            improved = True
                            break  # Restart inner loop
        
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
        
            if time.time() > end_time:
                break
        
            # Kick: remove 1-3 random elements from current_mask
            kick_size = rng.randint(1, 3)
            temp_mask = current_mask
            temp_size = current_size
            
            to_remove = []
            for _ in range(kick_size):
                if temp_size == 0:
                    break
                # Pick a random element from temp_mask
                bits = [i for i in range(n) if temp_mask & (1 << i)]
                if not bits:
                    break
                idx = rng.choice(bits)
                to_remove.append(idx)
                temp_mask &= ~(1 << idx)
                temp_size -= 1
            
            # Repair temp_mask
            # Try to add elements greedily
            for i in range(n):
                if not (temp_mask & (1 << i)):
                    if can_add_bitmask(temp_mask, i):
                        temp_mask |= (1 << i)
                        temp_size += 1
            
            # Accept if better or same with some probability
            if temp_size > current_size:
                current_mask = temp_mask
                current_size = temp_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            elif temp_size == current_size and rng.random() < 0.1:
                current_mask = temp_mask
                current_size = temp_size
        
        # Convert best_mask back to list
        result = [vals[i] for i in range(n) if best_mask & (1 << i)]
        
        # Final safety check
        if not is_sum_free(result):
            # Fallback to odd
            return [x for x in vals if x % 2 != 0]
        
        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        
        vals = instance(size, seed)
        
        # Allocate time
        if size <= 60:
            budget = 35.0
        elif size <= 120:
            budget = 50.0
        else:
            budget = 55.0
        
        remaining = DEADLINE - elapsed - 5
        budget = min(budget, remaining)
        
        result = solve_one(vals, time_budget=budget)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
