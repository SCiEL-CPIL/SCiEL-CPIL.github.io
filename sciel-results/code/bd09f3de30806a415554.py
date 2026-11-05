import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, 10s safety margin
    
    N = 17
    max_comparators = 4096
    
    def _now():
        return time.time()
    
    def _time_exceeded():
        return _now() > deadline
    
    def _apply_comparator(arr, i, j):
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
    
    def _verify_booleans_numpy(comparators, N, num_samples=10000):
        """Verify using random Boolean inputs with NumPy."""
        if not comparators:
            return True
        
        # Test with random Boolean inputs
        rng = np.random.RandomState(42)
        for _ in range(num_samples):
            arr = rng.randint(0, 2, size=N).astype(np.int32)
            for i, j in comparators:
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
            # Check if sorted
            if not np.all(arr[:-1] <= arr[1:]):
                return False
        return True
    
    def _verify_all_booleans(comparators, N):
        """Full verification over all 2^N Boolean inputs."""
        if not comparators:
            return True
        
        for mask in range(1 << N):
            arr = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int32)
            for i, j in comparators:
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
            if not np.all(arr[:-1] <= arr[1:]):
                return False
        return True
    
    def _count_comparators(comparators):
        return len(comparators)
    
    def _compute_depth(comparators):
        """Compute parallel depth of the network."""
        if not comparators:
            return 0
        in_degree = {}
        for i, j in comparators:
            in_degree[i] = in_degree.get(i, 0) + 1
            in_degree[j] = in_degree.get(j, 0) + 1
        # Use topological sort approach
        from collections import deque
        graph = {i: [] for i in range(N)}
        indeg = {i: 0 for i in range(N)}
        for idx, (i, j) in enumerate(comparators):
            graph[i].append((j, idx))
            graph[j].append((i, idx))
            indeg[i] += 1
            indeg[j] += 1
        
        # Simpler: just count layers by tracking which comparators can run in parallel
        # A comparator (i,j) can run in layer k if both i and j are free
        # This is complex; for now, approximate by assuming sequential
        return len(comparators)  # Conservative estimate
    
    def _odd_even_transposition(N):
        """Generate Odd-Even Transposition network."""
        comps = []
        for step in range(N):
            if step % 2 == 0:
                for i in range(0, N - 1, 2):
                    comps.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    comps.append((i, i + 1))
        return comps
    
    def _knuth_network(N):
        """Generate Knuth's sorting network (small N)."""
        # For N=17, use a known construction
        # This is a placeholder; we'll use odd-even as seed
        return _odd_even_transposition(N)
    
    def _build_from_layers(layers):
        """Convert list of layers (each layer is a set of non-conflicting comparators) to flat list."""
        comps = []
        for layer in layers:
            comps.extend(layer)
        return comps
    
    def _generate_random_layer(N, used_wires=None):
        """Generate a random layer of non-conflicting comparators."""
        if used_wires is None:
            used_wires = set()
        
        available = [i for i in range(N) if i not in used_wires]
        random.shuffle(available)
        
        layer = []
        for i in range(0, len(available) - 1, 2):
            a, b = available[i], available[i + 1]
            if a < b:
                layer.append((a, b))
            else:
                layer.append((b, a))
        return layer
    
    def _random_network(N, num_layers):
        """Generate a random network with given number of layers."""
        layers = []
        for _ in range(num_layers):
            layer = _generate_random_layer(N)
            layers.append(layer)
        return _build_from_layers(layers)
    
    def _mutate_network(comparators, N, mutation_rate=0.1):
        """Apply local search mutations: swap, insert, delete comparators."""
        if not comparators:
            return comparators
        
        new_comps = list(comparators)
        
        # Random swaps
        for _ in range(int(len(new_comps) * mutation_rate)):
            idx = random.randint(0, len(new_comps) - 1)
            i, j = new_comps[idx]
            # Try to swap with another wire
            k = random.randint(0, N - 1)
            while k == i or k == j:
                k = random.randint(0, N - 1)
            if k < i:
                new_comps[idx] = (k, i)
            elif k > j:
                new_comps[idx] = (j, k)
            else:
                new_comps[idx] = (i, k)
        
        # Random deletions
        for _ in range(int(len(new_comps) * mutation_rate * 0.5)):
            if len(new_comps) > 1:
                idx = random.randint(0, len(new_comps) - 1)
                new_comps.pop(idx)
        
        # Random insertions
        for _ in range(int(len(new_comps) * mutation_rate * 0.3)):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(new_comps))
            new_comps.insert(idx, (i, j))
        
        return new_comps
    
    def _greedy_remove(comparators, N, time_limit=5.0):
        """Try to remove comparators while maintaining correctness."""
        if not comparators:
            return comparators
        
        current = list(comparators)
        end_time = _now() + time_limit
        
        # Try removing each comparator
        changed = True
        while changed and _now() < end_time:
            changed = False
            for idx in range(len(current) - 1, -1, -1):
                if _now() > end_time:
                    break
                # Try removing comparator at idx
                candidate = current[:idx] + current[idx+1:]
                # Quick check with random samples
                if _verify_booleans_numpy(candidate, N, num_samples=500):
                    current = candidate
                    changed = True
        
        return current
    
    # Main algorithm
    best_network = None
    best_count = max_comparators
    
    # Start with Odd-Even Transposition
    seed_network = _odd_even_transposition(N)
    best_network = seed_network
    best_count = len(seed_network)
    
    # Genetic algorithm with local search
    population_size = 20
    generations = 50
    
    # Initialize population
    population = [seed_network]
    for _ in range(population_size - 1):
        net = _mutate_network(seed_network, N, mutation_rate=0.2)
        population.append(net)
    
    for gen in range(generations):
        if _time_exceeded():
            break
        
        # Evaluate fitness (comparator count)
        scored = []
        for net in population:
            count = len(net)
            scored.append((count, net))
        
        scored.sort(key=lambda x: x[0])
        
        # Keep best
        if scored[0][0] < best_count:
            best_count = scored[0][0]
            best_network = scored[0][1]
        
        # Selection: keep top half
        survivors = [net for _, net in scored[:population_size // 2]]
        
        # Crossover and mutation
        new_population = list(survivors)
        while len(new_population) < population_size:
            if _time_exceeded():
                break
            
            # Select two parents
            p1, p2 = random.sample(survivors, 2)
            
            # Crossover: take first half from p1, second half from p2
            mid = min(len(p1), len(p2)) // 2
            child = p1[:mid] + p2[mid:]
            
            # Mutate
            child = _mutate_network(child, N, mutation_rate=0.15)
            new_population.append(child)
        
        population = new_population
        
        # Periodic greedy removal on best
        if gen % 10 == 0 and _now() < deadline - 10:
            improved = _greedy_remove(best_network, N, time_limit=min(3.0, deadline - _now() - 5))
            if len(improved) < best_count:
                best_count = len(improved)
                best_network = improved
    
    # Final greedy removal with remaining time
    remaining_time = deadline - _now()
    if remaining_time > 2.0:
        final = _greedy_remove(best_network, N, time_limit=min(remaining_time - 1.0, 10.0))
        if len(final) < best_count:
            best_count = len(final)
            best_network = final
    
    # Final verification with more samples
    if _verify_booleans_numpy(best_network, N, num_samples=5000):
        return best_network
    
    # Fallback to seed if verification fails
    return _odd_even_transposition(N)
