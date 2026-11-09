import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def get_conflicts(S):
        """
        Returns a dictionary mapping each element in S to the number of conflicts it is involved in.
        A conflict is a triple (a,b,c) in S such that a+b=c.
        """
        s_set = set(S)
        conflicts = {x: 0 for x in S}
        # Count distinct triples? Or just count pairs (a,b) such that a+b in S?
        # If we count pairs, (a,b) and (b,a) are distinct if a!=b.
        # Let's count the number of pairs (a,b) in S x S such that a+b is in S.
        # This is a proxy for "how much this element contributes to violations".
        # Actually, we want to remove the element that breaks the most triples.
        # If we remove x, we break all triples containing x.
        # Number of triples containing x:
        # 1. x = a+b. Pairs (a,b) in S such that a+b=x.
        # 2. x = a. Pairs (b,c) in S such that x+b=c.
        # 3. x = b. Pairs (a,c) in S such that a+x=c.
        
        # Efficiently:
        # For each x in S:
        # count_x = 0
        # for y in S:
        #    if x+y in S: count_x += 1 (covers x as a or b, and x+y as c)
        #    if y-x in S: count_x += 1 (covers x as c, y as a, y-x as b) -> wait, y-x = a => y = a+x.
        #    if 2*x in S: count_x += 1 (covers x as a, x as b, 2x as c)
        
        # Let's just count pairs (a,b) in S such that a+b in S.
        # If we remove x, how many such pairs are destroyed?
        # A pair (a,b) is destroyed if a=x or b=x.
        # So we can just compute for each x:
        #   count = 0
        #   for y in S:
        #       if x+y in S: count += 1
        #       if y-x in S: count += 1 # y = a+x, so a=y-x. If a in S, then a+x=y in S.
        #   # Note: if x+y in S, it's counted in the loop for y.
        #   # If y-x in S, it's counted in the loop for y.
        #   # This double counts some?
        #   # Let's stick to a simpler metric: number of (y,z) in S such that x+y=z or x+z=y or y+z=x.
        
        s_list = list(S)
        n = len(s_list)
        # Precompute sum set for O(1) lookup? S is a set, so lookup is O(1).
        # Complexity O(N^2) per evaluation.
        # With N=240, N^2 = 57600. Very fast.
        
        for x in s_list:
            count = 0
            for y in s_list:
                if x == y:
                    if 2*x in s_set:
                        count += 1
                else:
                    if x + y in s_set:
                        count += 1
                    if y - x in s_set:
                        count += 1
            conflicts[x] = count
        return conflicts

    def greedy_repair(S, values, time_limit):
        """
        Greedily remove elements with highest conflict count until sum-free.
        """
        S = set(S)
        start_t = time.time()
        while True:
            if time.time() - start_t > time_limit:
                break
            if not S:
                break
            conflicts = get_conflicts(S)
            # Check if sum-free
            if all(c == 0 for c in conflicts.values()):
                break
            
            # Find element with max conflict
            max_conf = -1
            max_elem = None
            # Randomize tie-breaking to explore different paths
            candidates = [x for x, c in conflicts.items() if c == max(c.values())]
            if candidates:
                max_elem = random.choice(candidates)
                max_conf = conflicts[max_elem]
            
            if max_conf == 0:
                break
                
            S.remove(max_elem)
        
        return S

    def greedy_add_back(S, removed_elements, time_limit):
        """
        Try to add back removed elements if they don't create conflicts.
        """
        S = set(S)
        start_t = time.time()
        # Shuffle removed elements to avoid bias
        removed = list(removed_elements)
        random.shuffle(removed)
        
        for x in removed:
            if time.time() - start_t > time_limit:
                break
            if x in S:
                continue
            
            # Check if adding x creates a conflict
            # Conflict if:
            # 1. 2x in S
            # 2. exists y in S such that x+y in S
            # 3. exists y in S such that y-x in S (i.e., y = x + z for some z in S)
            
            is_conflict = False
            if 2*x in S:
                is_conflict = True
            else:
                for y in S:
                    if x + y in S:
                        is_conflict = True
                        break
                    if y - x in S:
                        is_conflict = True
                        break
            
            if not is_conflict:
                S.add(x)
        
        return S

    def solve_instance(values, budget):
        n = len(values)
        if n <= 2:
            return list(values)
        
        v_set = set(values)
        
        # Generate diverse starting points
        starts = []
        
        # 1. Odds
        starts.append([x for x in values if x % 2 != 0])
        
        # 2. Evens (if odds is small, evens might be better? No, evens are not sum-free usually, but let's see)
        # Actually, if we divide all evens by 2, we get a smaller problem.
        # But for now, let's just try structural splits.
        
        # 3. Top half by value
        mid = n // 2
        starts.append(values[mid:])
        
        # 4. Bottom half by value
        starts.append(values[:mid])
        
        # 5. Large positives
        starts.append([x for x in values if x > 0 and x > max(values)/2])
        
        # 6. Large negatives
        starts.append([x for x in values if x < 0 and x < min(values)/2])
        
        # 7. Random subsets
        for _ in range(5):
            starts.append(random.sample(values, n // 2))
        
        best_S = set()
        best_size = 0
        
        current_time = time.time()
        time_per_start = budget / len(starts)
        
        for start_list in starts:
            if time.time() - current_time > budget:
                break
                
            S_init = set(start_list)
            removed = v_set - S_init
            
            # Repair
            S_repaired = greedy_repair(S_init, values, time_per_start * 0.6)
            
            # Add back
            S_final = greedy_add_back(S_repaired, removed, time_per_start * 0.4)
            
            # Validate
            if is_sum_free(S_final):
                if len(S_final) > best_size:
                    best_size = len(S_final)
                    best_S = S_final
        
        if not best_S:
            # Fallback: Odds
            best_S = set(x for x in values if x % 2 != 0)
            # Ensure it's sum-free (it should be, but just in case of logic error in is_sum_free or generation)
            # Odds is always sum-free for integers?
            # a,b odd => a+b even. If S contains only odds, a+b (even) is not in S.
            # So yes, odds is always sum-free.
            
        return list(best_S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue

        # Allocate budget
        # Greedy repair is O(N^3) worst case (N removals, O(N^2) check each).
        # For N=240, 240^3 = 13M ops. Python is slow, maybe ~1s per full pass?
        # But we stop when sum-free.
        # Let's give more time to larger instances.
        budget = min(remaining - 2, 40 * (size / 240))
        budget = max(budget, 5)

        vals = instance(size, seed)
        res = solve_instance(vals, budget)

        if not is_sum_free(res):
            # Fallback
            res = [x for x in vals if x % 2 != 0]

        results.append(res)

    return results
