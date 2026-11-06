"""Combined program: efficient bad-triple detection + high-frequency randomized greedy + targeted local search."""
import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def _num_omega(n):
    """Count distinct prime factors."""
    if n <= 1:
        return 0
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 128
    universe = list(range(1, n + 1))
    universe_set = set(universe)

    # Precompute LCM table for all pairs
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = _lcm(i, j)

    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]

    # Efficiently find all bad triples
    # A triple (a,b,c) with a<b<c is bad iff lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Group pairs by their LCM value
    lcm_to_pairs = defaultdict(list)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_to_pairs[l].append((i, j))

    bad_triples = []
    for l, pairs in lcm_to_pairs.items():
        if len(pairs) < 2:
            continue
        # For this LCM value l, find triples (a,b,c) where all three pairwise LCMs equal l
        # This means a|l, b|l, c|l, and for each pair, lcm = l
        # i.e., the "complement" of each element within l (in terms of prime powers) 
        # must be covered by the other two elements
        
        # Simpler: just check all triples from the set of divisors of l that are in [1,128]
        candidates = [x for x in range(1, n + 1) if l % x == 0]
        if len(candidates) < 3:
            continue
        
        # For each pair in candidates, check if their LCM is l
        # Then for the third element, check if its LCM with both is l
        # Use the precomputed pairs list for efficiency
        # Build a set of pairs that have LCM exactly l
        pair_set = set()
        for (i, j) in pairs:
            pair_set.add((i, j))
        
        # For each element c in candidates, find pairs (a,b) with a<b<c such that
        # (a,b) in pair_set, (a,c) in pair_set, (b,c) in pair_set
        # Actually, let's just iterate over triples from candidates
        # But candidates could be large. Let's be smarter.
        
        # For each pair (a,b) in pair_set, find c > b such that (a,c) in pair_set and (b,c) in pair_set
        # Build adjacency: for each element x, which other elements y have lcm(x,y) = l?
        adj = defaultdict(set)
        for (i, j) in pair_set:
            adj[i].add(j)
            adj[j].add(i)
        
        # Now find triangles in this graph
        for a in candidates:
            neighbors_a = adj[a]
            for b in neighbors_a:
                if b <= a:
                    continue
                # Common neighbors of a and b that are > b
                common = neighbors_a & adj[b]
                for c in common:
                    if c > b:
                        bad_triples.append((a, b, c))

    # Build forbidden_pairs: for each x, set of (a,b) such that {a,b,x} is a bad triple
    forbidden_pairs = [set() for _ in range(n + 1)]
    for (a, b, c) in bad_triples:
        forbidden_pairs[c].add((a, b))
        forbidden_pairs[a].add((b, c))
        forbidden_pairs[b].add((a, c))

    # Convert to frozensets for faster hashing if needed, but tuples work fine
    # Precompute constraint degree
    constraint_degree = [0] * (n + 1)
    for x in range(1, n + 1):
        constraint_degree[x] = len(forbidden_pairs[x])

    def can_add(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        for pair in forbidden_pairs[x]:
            a, b = pair
            if a in selected_set and b in selected_set:
                return False
        return True

    def greedy(order, selected_set=None):
        """Greedy construction with given order."""
        if selected_set is None:
            selected_set = set()
        else:
            selected_set = set(selected_set)
        selected = list(selected_set)
        for x in order:
            if x not in selected_set and can_add(x, selected_set):
                selected.append(x)
                selected_set.add(x)
        return selected

    def make_order(seed, strategy):
        """Generate element ordering."""
        random.seed(seed)
        if strategy == 0:
            # Natural order
            return list(universe)
        elif strategy == 1:
            # By number of prime factors (ascending)
            return sorted(universe, key=lambda x: (_num_omega(x), x))
        elif strategy == 2:
            # Random
            order = list(universe)
            random.shuffle(order)
            return order
        elif strategy == 3:
            # Reverse natural
            return list(reversed(universe))
        elif strategy == 4:
            # By value descending, then omega
            return sorted(universe, key=lambda x: (-_num_omega(x), x))
        elif strategy == 5:
            # Random with bias toward low omega
            weights = [1.0 / (1 + _num_omega(x)) for x in universe]
            order = []
            remaining = list(universe)
            remaining_weights = weights[:]
            while remaining:
                total = sum(remaining_weights)
                r = random.random() * total
                cum = 0
                for idx, w in enumerate(remaining_weights):
                    cum += w
                    if cum >= r:
                        order.append(remaining[idx])
                        remaining.pop(idx)
                        remaining_weights.pop(idx)
                        break
            return order
        elif strategy == 6:
            # By constraint degree (ascending) - less constrained first
            return sorted(universe, key=lambda x: (constraint_degree[x], x))
        elif strategy == 7:
            # By constraint degree (descending) - more constrained first
            return sorted(universe, key=lambda x: (-constraint_degree[x], x))
        elif strategy == 8:
            # Evens first
            evens = [x for x in universe if x % 2 == 0]
            odds = [x for x in universe if x % 2 == 1]
            random.shuffle(evens)
            random.shuffle(odds)
            return evens + odds
        elif strategy == 9:
            # Odds first
            odds = [x for x in universe if x % 2 == 1]
            evens = [x for x in universe if x % 2 == 0]
            random.shuffle(odds)
            random.shuffle(evens)
            return odds + evens
        return list(universe)

    def local_search_kick(initial_set, max_iterations=3000, kick_min=2, kick_max=5):
        """Local search with kick-and-refill strategy."""
        selected_set = set(initial_set)
        best_size = len(selected_set)
        best_set = set(selected_set)
        all_elements = list(universe)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            # Determine kick size
            kick_size = random.randint(kick_min, kick_max)
            if kick_size > len(selected_set):
                kick_size = max(1, len(selected_set) - 1)
            
            if kick_size == 0:
                continue
                
            # Select elements to remove
            removed = random.sample(list(selected_set), kick_size)
            remaining_set = selected_set - set(removed)
            
            # Candidates to add (everything not in remaining_set)
            candidates = [x for x in all_elements if x not in remaining_set]
            random.shuffle(candidates)
            
            # Greedily add
            new_set = set(remaining_set)
            for x in candidates:
                if can_add(x, new_set):
                    new_set.add(x)
            
            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = set(new_set)
                selected_set = new_set
            elif len(new_set) == len(selected_set):
                # Accept equal-size moves with some probability to escape local optima
                if random.random() < 0.3:
                    selected_set = new_set
        
        return list(best_set)

    def local_search_swap(initial_set, max_iterations=5000):
        """Local search with 1-swap and 1-add moves."""
        selected_set = set(initial_set)
        best_size = len(selected_set)
        best_set = set(selected_set)
        all_elements = list(universe)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            move_type = random.randint(0, 2)
            new_set = set(selected_set)
            
            if move_type == 0:  # 1-add
                candidates = [x for x in all_elements if x not in selected_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if can_add(x, selected_set):
                    new_set.add(x)
                else:
                    continue
            elif move_type == 1:  # 1-remove (rarely helps in maximization, but can enable future adds)
                if not selected_set:
                    continue
                x = random.choice(list(selected_set))
                new_set.remove(x)
            else:  # 2-swap: remove one, add one
                if not selected_set:
                    continue
                to_remove = random.choice(list(selected_set))
                candidates = [x for x in all_elements if x not in selected_set]
                if not candidates:
                    continue
                to_add = random.choice(candidates)
                new_set.discard(to_remove)
                if can_add(to_add, new_set):
                    new_set.add(to_add)
                else:
                    new_set.add(to_remove)
                    continue
            
            delta = len(new_set) - len(selected_set)
            
            # Accept if better, or with probability if equal/worse
            if delta >= 0:
                selected_set = new_set
                if len(selected_set) > best_size:
                    best_size = len(selected_set)
                    best_set = set(selected_set)
            elif random.random() < 0.1:  # small probability of accepting worse moves
                selected_set = new_set
        
        return list(best_set)

    # Generate multiple orderings and run greedy
    best_result = []
    best_size = 0

    # Deterministic strategies
    deterministic_strategies = [0, 1, 3, 4, 6, 7, 8, 9]
    for s in deterministic_strategies:
        if time.time() - start_time > DEADLINE:
            break
        order = make_order(0, s)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Randomized greedy passes - many of them
    random_seed_base = 12345
    num_random_passes = 200
    for i in range(num_random_passes):
        if time.time() - start_time > DEADLINE - 10:
            break
        seed = random_seed_base + i * 7919
        # Mix of random and biased strategies
        if i % 3 == 0:
            s = 2  # pure random
        elif i % 3 == 1:
            s = 5  # biased toward low omega
        else:
            s = 2  # random again
        order = make_order(seed, s)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Local search on the best result
    if time.time() - start_time < DEADLINE - 5:
        improved = local_search_kick(best_result, max_iterations=5000, kick_min=2, kick_max=5)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = improved

    # Additional local search on other good results
    if time.time() - start_time < DEADLINE - 8:
        # Collect some good solutions from greedy
        good_solutions = []
        for i in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            seed = 999999 + i * 31337
            order = make_order(seed, 2)
            result = greedy(order)
            if len(result) >= best_size - 3:
                good_solutions.append(result)
        
        for sol in good_solutions[:10]:
            if time.time() - start_time > DEADLINE - 3:
                break
            improved = local_search_kick(sol, max_iterations=3000, kick_min=2, kick_max=4)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = improved

    # Swap-based local search on best
    if time.time() - start_time < DEADLINE - 3:
        improved = local_search_swap(best_result, max_iterations=5000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = improved

    # Final polish with smaller kicks
    if time.time() - start_time < DEADLINE - 1:
        improved = local_search_kick(best_result, max_iterations=3000, kick_min=1, kick_max=3)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = improved

    # Ensure nonempty
    if not best_result:
        best_result = [1]

    return best_result
