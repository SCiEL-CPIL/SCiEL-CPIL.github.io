"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def build_conflict_graph(S):
        """Build adjacency matrix for conflict graph.
        Two elements conflict if they form a sum triple with some element in S.
        i.e., x and y conflict if x+y in S, or 2x in S (self-conflict).
        For the independent set interpretation: we want a set where no two elements
        have their sum in the set. But actually the constraint is on triples within
        the selected set. Let me reconsider.
        
        The constraint: no a, b, c in T with a+b=c.
        This is a 3-uniform hypergraph constraint, not a graph constraint.
        However, we can model it differently: if we select a and b, then a+b cannot be selected.
        So for a fixed pair (a,b), the element a+b is forbidden.
        
        For local search, we can track: for each element x, which pairs (a,b) with a+b=x
        are both selected. If any such pair exists, x must be removed.
        
        Let's build: for each x, parents[x] = list of (a,b) where a+b=x and a,b in S.
        """
        S_set = set(S)
        n = len(S)
        idx_map = {x: i for i, x in enumerate(S)}
        
        # For each element, find all pairs that sum to it
        parents = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in S_set:
                    k = idx_map[s]
                    parents[k].append((i, j))
        
        return parents

    def can_add(x, current_set, S_set, idx_map, parents_by_value):
        """Check if x can be added to current_set."""
        if 2 * x in current_set:
            return False
        for a in current_set:
            if a + x in current_set:
                return False
            if x - a in current_set:
                return False
        return True

    def greedy_initial(S, order_key):
        """Greedy construction with given ordering."""
        S_sorted = sorted(S, key=order_key)
        result = []
        result_set = set()
        for x in S_sorted:
            if time.time() > DEADLINE - 5:
                break
            if can_add(x, result_set, set(S), None, None):
                result.append(x)
                result_set.add(x)
        return result

    def local_search_with_swaps(S, initial_set, time_budget=20.0):
        """Local search with add, remove, and swap operations."""
        current = set(initial_set)
        best = set(current)
        best_size = len(current)
        S_set = set(S)
        
        start_time = time.time()
        
        while time.time() < start_time + time_budget and time.time() < DEADLINE - 10:
            improved = False
            
            # Try adding each candidate
            for x in S:
                if x in current:
                    continue
                if can_add(x, current, S_set, None, None):
                    current.add(x)
                    improved = True
            
            # Try swaps: remove one, add another
            if not improved:
                for a in list(current):
                    for x in S:
                        if x in current:
                            continue
                        temp = set(current)
                        temp.discard(a)
                        if can_add(x, temp, S_set, None, None):
                            temp.add(x)
                            if len(temp) > len(best):
                                best = set(temp)
                                best_size = len(best)
                            current = temp
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing elements that block many additions
            if not improved:
                for a in list(current):
                    temp = set(current)
                    temp.discard(a)
                    addable = 0
                    for x in S:
                        if x in temp:
                            continue
                        if can_add(x, temp, S_set, None, None):
                            addable += 1
                    if addable > 1:
                        current = temp
                        improved = True
            
            if len(current) > best_size:
                best = set(current)
                best_size = len(best)
            
            if not improved:
                break
        
        return list(best)

    def simulated_annealing(S, initial_set, time_budget=20.0):
        """Simulated annealing for sum-free subset optimization."""
        current = set(initial_set)
        best = set(current)
        best_size = len(current)
        S_set = set(S)
        
        start_time = time.time()
        T = 10.0  # initial temperature
        
        while time.time() < start_time + time_budget and time.time() < DEADLINE - 10:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try adding a random element
                candidates = [x for x in S if x not in current]
                if candidates:
                    x = random.choice(candidates)
                    if can_add(x, current, S_set, None, None):
                        current.add(x)
                        if len(current) > best_size:
                            best = set(current)
                            best_size = len(best)
                    else:
                        # Move rejected, maybe accept with probability
                        if random.random() < np.exp(-1.0 / max(T, 0.01)):
                            current.add(x)  # This would violate, so don't do this
                        # Actually, we can't add if it violates. So skip.
                        pass
            
            elif move_type < 0.7:
                # Try removing a random element
                if current:
                    a = random.choice(list(current))
                    current.discard(a)
            
            else:
                # Try a swap
                if current:
                    a = random.choice(list(current))
                    candidates = [x for x in S if x not in current]
                    if candidates:
                        x = random.choice(candidates)
                        temp = set(current)
                        temp.discard(a)
                        if can_add(x, temp, S_set, None, None):
                            temp.add(x)
                            if len(temp) > best_size:
                                best = set(temp)
                                best_size = len(best)
                            current = temp
            
            # Cool down
            T *= 0.999
        
        return list(best)

    def solve_instance(S, time_budget=30.0):
        """Solve one instance with multiple strategies."""
        n = len(S)
        
        best = None
        best_size = 0
        
        # Multiple greedy strategies
        strategies = [
            lambda x: x % 2 != 0,  # Odds
            lambda x: (-abs(x), x),  # Large |x| first
            lambda x: (abs(x), x),  # Small |x| first
            lambda x: (-x, abs(x)),  # Positive first
            lambda x: (x, abs(x)),  # Negative first
            lambda x: random.Random(hash(x)).random(),  # Random
        ]
        
        for key_func in strategies:
            if time.time() > DEADLINE - 15:
                break
            
            initial = greedy_initial(S, key_func)
            
            if len(initial) > best_size:
                best = initial
                best_size = len(initial)
            
            # Local search improvement
            if time.time() < DEADLINE - 20:
                remaining = min(15, DEADLINE - 20 - time.time())
                if remaining > 5:
                    improved = local_search_with_swaps(S, initial, time_budget=remaining)
                    if len(improved) > best_size:
                        best = improved
                        best_size = len(improved)
        
        # Simulated annealing from best
        if best and time.time() < DEADLINE - 15:
            remaining = min(15, DEADLINE - 15 - time.time())
            if remaining > 5:
                sa_result = simulated_annealing(S, best, time_budget=remaining)
                if len(sa_result) > best_size:
                    best = sa_result
                    best_size = len(best_size)
        
        # Random restarts
        rng = random.Random(123)
        for _ in range(5):
            if time.time() > DEADLINE - 15:
                break
            S_shuffled = S[:]
            rng.shuffle(S_shuffled)
            initial = []
            init_set = set()
            for x in S_shuffled:
                if can_add(x, init_set, set(S), None, None):
                    initial.append(x)
                    init_set.add(x)
            
            if len(initial) > best_size:
                best = initial
                best_size = len(initial)
            
            if time.time() < DEADLINE - 20:
                remaining = min(8, DEADLINE - 20 - time.time())
                if remaining > 3:
                    improved = local_search_with_swaps(S, initial, time_budget=remaining)
                    if len(improved) > best_size:
                        best = improved
                        best_size = len(improved)
        
        if best is None:
            best = [x for x in S if x % 2 != 0]
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Allocate time based on instance size
        if size <= 100:
            budget = 25.0
        elif size <= 200:
            budget = 40.0
        else:
            budget = 55.0
        
        best = solve_instance(S, time_budget=budget)
        
        # Final verification
        if not is_sum_free(best):
            best = [x for x in S if x % 2 != 0]
        
        results.append(best)
    
    return results
# EVOLVE-BLOCK-END
