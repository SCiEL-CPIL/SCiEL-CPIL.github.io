import random
import math
import time

N = 64

def construct():
    """Construct a set of grid points in [0, N)^2 with no three forming an isosceles triangle."""
    start_time = time.time()
    deadline = start_time + 310  # Safety margin before 330s kill

    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_index = {p: i for i, p in enumerate(all_points)}

    # Precompute squared distances between all pairs
    n_pts = N * N
    dist_sq = [[0] * n_pts for _ in range(n_pts)]
    for i in range(n_pts):
        xi, yi = all_points[i]
        for j in range(i + 1, n_pts):
            xj, yj = all_points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist_sq[i][j] = d
            dist_sq[j][i] = d

    def check_validity(points):
        """Check if a set of points has no isosceles triangle. O(k^2) but with early exit."""
        k = len(points)
        if k < 3:
            return True
        for i in range(k):
            pi = points[i]
            seen_dists = set()
            for j in range(k):
                if i == j:
                    continue
                pj = points[j]
                d = (pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2
                if d in seen_dists:
                    return False
                seen_dists.add(d)
        return True

    def fast_check_addition(point, current_indices, dist_cache):
        """
        Check if adding 'point' to the set maintains no-isosceles property.
        Uses precomputed distances. O(k) per check.
        """
        pi = point_index[point]
        k = len(current_indices)
        seen_dists = set()
        
        # Check distances from new point to all existing points
        for j in range(k):
            pj_idx = current_indices[j]
            d = dist_sq[pi][pj_idx]
            if d in seen_dists:
                return False
            seen_dists.add(d)
        
        # Check if any existing pair has a distance equal to one of the new distances
        # This is the tricky part - we need to check all pairs
        # But we can optimize: for each existing point j, the distance from j to new point
        # must not equal distance from j to any other existing point
        for j in range(k):
            pj_idx = current_indices[j]
            d_new = dist_sq[pi][pj_idx]
            for l in range(k):
                if l == j:
                    continue
                pl_idx = current_indices[l]
                d_existing = dist_sq[pj_idx][pl_idx]
                if d_new == d_existing:
                    return False
        
        return True

    def greedy_build(seed, max_time=None):
        """Build a set greedily using random ordering."""
        rng = random.Random(seed)
        points = list(all_points)
        rng.shuffle(points)
        
        current = []
        current_indices = []
        
        for p in points:
            if max_time and time.time() > max_time:
                break
            if fast_check_addition(p, current_indices, None):
                current.append(p)
                current_indices.append(point_index[p])
        
        return current

    def local_search(initial_set, seed, max_time=None):
        """Improve the set via local search with add operations."""
        rng = random.Random(seed)
        current = list(initial_set)
        current_set = set(current)
        current_indices = [point_index[p] for p in current]
        
        def try_add(point):
            nonlocal current, current_set, current_indices
            if point not in current_set:
                if fast_check_addition(point, current_indices, None):
                    current.append(point)
                    current_set.add(point)
                    current_indices.append(point_index[point])
                    return True
            return False
        
        all_pts = list(all_points)
        improved = True
        while improved and (max_time is None or time.time() < max_time):
            improved = False
            rng.shuffle(all_pts)
            for p in all_pts:
                if max_time and time.time() > max_time:
                    break
                if try_add(p):
                    improved = True
        
        return current

    # Strategy 1: Multiple greedy builds with different seeds
    best_set = []
    best_size = 0
    
    for seed in range(15):
        if time.time() >= deadline - 10:
            break
        s = greedy_build(seed, max_time=deadline - 15)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Strategy 2: Local search from greedy results
    for seed in range(20):
        if time.time() >= deadline - 10:
            break
        s = greedy_build(seed, max_time=deadline - 20)
        improved = local_search(s, seed, max_time=deadline - 10)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Strategy 3: Genetic algorithm
    def genetic_algorithm(pop_size, generations, max_time=None):
        """Evolve sets via crossover and mutation."""
        rng = random.Random(42)
        
        # Initialize population with greedy solutions
        population = []
        for i in range(min(pop_size, 8)):
            if max_time and time.time() > max_time:
                break
            s = greedy_build(i, max_time=max_time)
            population.append(list(s))
        
        # Fill remaining with random small sets
        while len(population) < pop_size:
            s = []
            for _ in range(rng.randint(3, 8)):
                p = rng.choice(all_points)
                if p not in s and fast_check_addition(p, [point_index[x] for x in s], None):
                    s.append(p)
            population.append(s)
        
        best_in_ga = max(population, key=len)
        
        for gen in range(generations):
            if max_time and time.time() > max_time:
                break
            
            # Evaluate fitness
            population.sort(key=len, reverse=True)
            
            if len(population[0]) > len(best_in_ga):
                best_in_ga = population[0][:]
            
            # Selection: keep top half
            survivors = population[:pop_size // 2]
            
            # Generate offspring
            new_pop = list(survivors)
            while len(new_pop) < pop_size:
                if max_time and time.time() > max_time:
                    break
                
                # Crossover: combine two parents
                p1 = rng.choice(survivors)
                p2 = rng.choice(survivors)
                
                # Start with larger parent
                child = list(p1)
                child_set = set(child)
                child_indices = [point_index[x] for x in child]
                
                # Try adding points from other parent
                for p in p2:
                    if p not in child_set:
                        if fast_check_addition(p, child_indices, None):
                            child.append(p)
                            child_set.add(p)
                            child_indices.append(point_index[p])
                
                # Mutation: try adding random points
                for _ in range(5):
                    p = rng.choice(all_points)
                    if p not in child_set:
                        if fast_check_addition(p, child_indices, None):
                            child.append(p)
                            child_set.add(p)
                            child_indices.append(point_index[p])
                
                new_pop.append(child)
            
            population = new_pop
        
        return best_in_ga
    
    # Run genetic algorithm with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        ga_result = genetic_algorithm(16, 20, max_time=deadline - 5)
        if len(ga_result) > best_size:
            best_set = ga_result
            best_size = len(ga_result)
    
    return best_set
