"""Maximum clique-style approach for 3-term progression differences."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_set_bitset(s):
    """
    Compute the number of distinct d > 0 such that a, a+d, a+2d all in s.
    Uses Python integers as bitsets for fast checking.
    """
    s_set = set(s)
    if len(s_set) < 3:
        return 0
    max_val = max(s_set)
    bitset = 0
    for x in s_set:
        bitset |= (1 << x)
    
    count = 0
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            count += 1
    return count


def _compute_diffs_bitset(s):
    """Return the set of d values that are realized."""
    s_set = set(s)
    if len(s_set) < 3:
        return set()
    max_val = max(s_set)
    bitset = 0
    for x in s_set:
        bitset |= (1 << x)
    
    diffs = set()
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            diffs.add(d)
    return diffs


def _max_clique_branch_and_bound(adj, vertices):
    """
    Find a maximum clique in a graph using branch and bound.
    adj: list of sets, adj[i] contains j if edge (i,j) exists
    vertices: list of vertex indices to consider
    Returns the maximum clique as a list of vertex indices.
    """
    best_clique = []
    
    def extend(clique, candidates):
        nonlocal best_clique
        if not candidates:
            if len(clique) > len(best_clique):
                best_clique = list(clique)
            return
        
        # Upper bound: |clique| + |candidates|
        if len(clique) + len(candidates) <= len(best_clique):
            return
        
        # Order candidates by degree in the subgraph (greedy coloring bound is complex, use simple)
        # Try vertices in order of highest connectivity first
        for i in range(len(candidates)):
            v = candidates[i]
            new_clique = clique + [v]
            # New candidates: those connected to v
            new_cands = [c for c in candidates[i+1:] if c in adj[v]]
            extend(new_clique, new_cands)
    
    extend([], list(vertices))
    return best_clique


def _build_diff_graph(target_size=32, max_val=512):
    """
    Build a graph where vertices are d values, and edges connect pairs
    that can coexist in a set of target_size elements.
    
    Two differences d1, d2 can coexist if there's a set of <= target_size
    elements that realizes both. We approximate this by checking if the
    minimal set realizing both has size <= target_size.
    """
    max_d = max_val // 2  # 256
    d_range = list(range(1, max_d + 1))
    n = len(d_range)
    
    # Precompute minimal set size needed for each pair (d1, d2)
    # A 3-AP with difference d uses 3 elements: a, a+d, a+2d
    # For two differences d1, d2, we need a set that contains at least
    # one AP with diff d1 and one AP with diff d2.
    # Minimum set size is 3 (if the two APs share elements) or 6 (if disjoint)
    # But we can always do it in <= 6 elements, which is <= 32, so all pairs
    # are feasible. The real constraint is fitting in [1, 512].
    
    # Let's build a weighted graph instead: weight(d1, d2) = how "easy" it is
    # to realize both, measured by the minimum span needed.
    
    # For a single d, the AP a, a+d, a+2d needs span 2d, so a can be at most 512-2d.
    # For two d's, we need to fit both APs in [1, 512].
    
    # Let's use a different approach: score each d by how many positions it has,
    # and build edges based on compatibility.
    
    adj = [set() for _ in range(n)]
    for i in range(n):
        d1 = d_range[i]
        for j in range(i + 1, n):
            d2 = d_range[j]
            # Check if both can fit: need 2*d1 <= 511 and 2*d2 <= 511, which is always true for d <= 255
            # The real question is whether we can choose positions to minimize overlap conflicts
            # For now, connect all pairs (complete graph) and let the search figure it out
            # This makes the clique problem trivial (take all vertices), so we need a better model.
            
            # Better model: the "cost" of including d is that it constrains the set.
            # Let's use a different formulation: instead of clique, do a weighted
            # independent set or a different combinatorial optimization.
            
            # Actually, let me reconsider. The real constraint is that we have only 32 elements.
            # Each d requires at least 3 specific elements. If d1 and d2 require the same
            # 3 elements (impossible for different d's unless they're the same), or if
            # they share some elements, the total set size is less than 6.
            
            # For two different differences d1 != d2, can they share elements?
            # AP1: a, a+d1, a+2d1
            # AP2: b, b+d2, b+2d2
            # They can share elements if some element from AP1 equals some from AP2.
            # E.g., a+d1 = b+d2, etc.
            
            # This is getting complex. Let me use a different approach entirely.
            pass
    
    # Abandon the pure clique approach; use a hybrid with local search but
    # with a smarter initial solution and perturbation strategy.
    return None


def _construct_simulated_annealing():
    """
    Simulated annealing to find a good set.
    """
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    random.seed(42)
    
    # Start with a dense interval
    current_set = list(range(1, 33))
    current_score = _score_set_bitset(current_set)
    
    best_set = list(current_set)
    best_score = current_score
    
    temperature = 10.0
    cooling_rate = 0.995
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Generate a neighbor: swap one element
        elem_to_remove = random.choice(current_set)
        new_elem = random.randint(1, 512)
        while new_elem in current_set:
            new_elem = random.randint(1, 512)
        
        new_set = set(current_set)
        new_set.remove(elem_to_remove)
        new_set.add(new_elem)
        new_score = _score_set_bitset(new_set)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / max(temperature, 1e-10)):
            current_set = list(new_set)
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(current_set)
        
        temperature *= cooling_rate
    
    return best_set


def _construct_genetic_local_search():
    """
    Combine genetic algorithm with local search.
    """
    start_time = time.time()
    deadline = 140.0
    random.seed(123)
    
    pop_size = 8
    # Initialize population with diverse sets
    population = []
    
    # Set 1: dense interval
    population.append(list(range(1, 33)))
    # Set 2: even numbers
    population.append(list(range(2, 65, 2)))
    # Set 3: random
    while len(population) < pop_size:
        s = set(random.sample(range(1, 513), 32))
        population.append(list(s))
    
    best_set = max(population, key=lambda x: _score_set_bitset(x))
    best_score = _score_set_bitset(best_set)
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Evaluate population
        scored = [( _score_set_bitset(s), s) for s in population]
        scored.sort(key=lambda x: -x[0])
        
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_set = list(scored[0][1])
        
        # Selection: keep top half
        survivors = [s for _, s in scored[:pop_size // 2]]
        
        # Crossover: create offspring
        offspring = []
        while len(offspring) < pop_size - len(survivors):
            if len(survivors) < 2:
                break
            p1, p2 = random.sample(survivors, 2)
            # Uniform crossover
            child = []
            set1, set2 = set(p1), set2
            for elem in set1 | set2:
                if elem in set1 and random.random() < 0.5:
                    child.append(elem)
                elif elem in set2 and random.random() < 0.5:
                    child.append(elem)
            
            # Fix size
            if len(child) < 32:
                while len(child) < 32:
                    e = random.randint(1, 512)
                    if e not in child:
                        child.append(e)
            elif len(child) > 32:
                child = random.sample(child, 32)
            
            # Ensure uniqueness
            child = list(set(child))
            while len(child) < 32:
                e = random.randint(1, 512)
                if e not in child:
                    child.append(e)
            
            offspring.append(child)
        
        # Mutation
        for i in range(len(offspring)):
            if random.random() < 0.5:
                idx = random.randint(0, 31)
                offspring[i][idx] = random.randint(1, 512)
                # Ensure uniqueness
                s = set(offspring[i])
                if len(s) < 32:
                    while len(s) < 32:
                        s.add(random.randint(1, 512))
                    offspring[i] = list(s)
        
        # Local search on best offspring
        population = survivors + offspring
        
        # Apply local search to the best member
        best_member = max(population, key=lambda x: _score_set_bitset(x))
        current_set = set(best_member)
        current_score = _score_set_bitset(current_set)
        
        # Quick local search: try swaps
        for _ in range(20):
            if time.time() > deadline:
                break
            improved = False
            for i in range(32):
                elem = list(current_set)[i]
                new_elem = random.randint(1, 512)
                if new_elem in current_set:
                    continue
                new_set = current_set - {elem} | {new_elem}
                new_score = _score_set_bitset(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    break
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
        
        # Replace worst members with locally searched result
        population = [list(current_set)] + population[1:]
    
    return best_set


def _construct_iterated_local_search():
    """
    Iterated Local Search with smart perturbation.
    """
    start_time = time.time()
    deadline = 140.0
    random.seed(42)
    
    # Start with dense interval
    current_set = set(range(1, 33))
    current_score = _score_set_bitset(current_set)
    
    best_set = list(current_set)
    best_score = current_score
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Local search: try all single-element swaps
        improved = True
        while improved:
            improved = False
            current_list = sorted(current_set)
            
            for i in range(len(current_list)):
                removed = current_list[i]
                # Try several random new elements rather than all 512
                for _ in range(20):
                    new_elem = random.randint(1, 512)
                    if new_elem in current_set:
                        continue
                    
                    new_set = current_set - {removed} | {new_elem}
                    new_score = _score_set_bitset(new_set)
                    
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best_set = list(current_set)
                        break
                if improved:
                    break
        
        # Perturbation: change 2-4 elements
        num_perturb = random.randint(2, 4)
        current_set = set(best_set)
        for _ in range(num_perturb):
            to_remove = random.choice(list(current_set))
            current_set.remove(to_remove)
            new_elem = random.randint(1, 512)
            while new_elem in current_set:
                new_elem = random.randint(1, 512)
            current_set.add(new_elem)
        
        current_score = _score_set_bitset(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
    
    return best_set


def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Run multiple strategies and take the best
    results = []
    
    # Strategy 1: Simulated annealing
    if time.time() < deadline - 40:
        sa_set = _construct_simulated_annealing()
        results.append(sa_set)
    
    # Strategy 2: Iterated local search
    if time.time() < deadline - 20:
        ils_set = _construct_iterated_local_search()
        results.append(ils_set)
    
    # Take the best
    if results:
        best = max(results, key=lambda s: _score_set_bitset(s))
        return best
    else:
        return list(range(1, 33))

# EVOLVE-BLOCK-END
