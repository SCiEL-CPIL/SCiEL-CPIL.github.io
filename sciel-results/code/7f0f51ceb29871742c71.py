<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin: finish well before 90s limit
    
    # Start with a known valid network using odd-even merge sort
    def oe_merge_sort(n):
        """Odd-even merge sort network for n elements."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = oe_merge_sort(half)
        right = oe_merge_sort(n - half)
        right = [(i + half, j + half) for i, j in right]
        
        # Odd-even merge of two sorted halves
        merge = []
        for phase in range(n - 1):
            offset = phase % 2
            for i in range(offset, n - 1, 2):
                merge.append((i, i + 1))
        
        return left + right + merge
    
    seed_network = oe_merge_sort(N)
    # Verify seed is valid
    if not is_sorted_network_full(seed_network):
        # Fallback to odd-even transposition
        seed_network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                seed_network.append((i, i + 1))
    
    best_network = list(seed_network)
    best_len = len(best_network)
    current = list(best_network)
    
    # Helper: try to remove comparator at index idx
    def try_remove(idx):
        nonlocal current, best_network, best_len
        candidate = current.copy()
        candidate.pop(idx)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
                return True
        return False
    
    # Phase 1: Aggressive systematic deletion - multiple passes
    for pass_num in range(50):
        if time.time() > deadline - 10.0:
            break
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 10.0:
                break
            if try_remove(i):
                progress = True
                # Don't increment i since list shifted
            else:
                i += 1
        if not progress:
            break
    
    # Phase 2: Random deletion with restarts
    rng_search = random.Random(12345)
    for restart in range(10):
        if time.time() > deadline - 6.0:
            break
        attempts = 0
        while time.time() < deadline - 6.0 and attempts < 500:
            attempts += 1
            if not current:
                break
            idx = rng_search.randint(0, len(current) - 1)
            if try_remove(idx):
                pass
            else:
                # Occasionally try inserting a new comparator then removing others
                if attempts % 50 == 0 and len(current) > 60:
                    # Try to find a useful insertion
                    i = rng_search.randint(0, N - 2)
                    j = rng_search.randint(i + 1, N - 1)
                    candidate = current.copy()
                    candidate.append((i, j))
                    if is_sorted_network_sample(candidate):
                        if is_sorted_network_full(candidate):
                            # Now try to remove some comparators
                            for k in range(len(candidate)):
                                if time.time() > deadline - 6.0:
                                    break
                                if try_remove(k):
                                    break
    
    # Phase 3: Local search - try moving comparators to different positions
    if time.time() < deadline - 4.0:
        for _ in range(500):
            if time.time() > deadline - 4.0:
                break
            if len(current) < 2:
                break
            idx = rng_search.randint(0, len(current) - 1)
            new_idx = rng_search.randint(0, len(current) - 1)
            if idx == new_idx:
                continue
            
            candidate = current.copy()
            comp = candidate.pop(idx)
            candidate.insert(new_idx, comp)
            
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    current = candidate
                    best_network = candidate
                    best_len = len(current)
    
    # Phase 4: Try replacing comparators with different ones
    if time.time() < deadline - 3.0:
        for _ in range(300):
            if time.time() > deadline - 3.0:
                break
            if not current:
                break
            idx = rng_search.randint(0, len(current) - 1)
            i = rng_search.randint(0, N - 2)
            j = rng_search.randint(i + 1, N - 1)
            if (i, j) == current[idx]:
                continue
            
            candidate = current.copy()
            candidate[idx] = (i, j)
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    current = candidate
                    best_network = candidate
                    best_len = len(current)
    
    # Phase 5: Final aggressive deletion pass
    i = 0
    while i < len(current) and time.time() < deadline - 1.0:
        if try_remove(i):
            pass
        else:
            i += 1
    
    # Final Check: Ensure we return a valid network
    if not is_sorted_network_full(best_network):
        return seed_network
        
    return best_network
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin: finish well before 90s limit

    # --- Layer-based representation ---
    # A network is represented as a list of layers, each layer is a set of
    # non-conflicting comparators (no two comparators share a wire).
    # This enables crossover: we can combine layers from different parents.

    def to_layers(comparators):
        """Convert a flat list of comparators to a list of layers (maximal parallel sets)."""
        layers = []
        current_layer = []
        used_wires = set()
        for comp in comparators:
            i, j = comp
            if i not in used_wires and j not in used_wires:
                current_layer.append(comp)
                used_wires.add(i)
                used_wires.add(j)
            else:
                if current_layer:
                    layers.append(current_layer)
                current_layer = [comp]
                used_wires = {i, j}
        if current_layer:
            layers.append(current_layer)
        return layers

    def from_layers(layers):
        """Convert layers back to a flat list of comparators."""
        comps = []
        for layer in layers:
            comps.extend(layer)
        return comps

    def simulate_layers(layers, input_matrix):
        """Simulate a layer-based network on input matrix."""
        state = input_matrix.copy()
        for layer in layers:
            for i, j in layer:
                a = state[:, i]
                b = state[:, j]
                mask = a > b
                state[:, i] = np.where(mask, b, a)
                state[:, j] = np.where(mask, a, b)
        return state

    def count_comparators(layers):
        return sum(len(layer) for layer in layers)

    def is_valid_network(layers):
        """Check if layer-based network sorts all inputs."""
        total = count_comparators(layers)
        if total > 4096:
            return False
        state = simulate_layers(layers, _INPUTS)
        return np.array_equal(state, _SORTED_INPUTS)

    def is_valid_sample(layers):
        """Fast check on sample."""
        total = count_comparators(layers)
        if total > 4096:
            return False
        state = simulate_layers(layers, _SAMPLE_INPUTS)
        return np.array_equal(state, _SAMPLE_SORTED)

    # --- Generate initial seed networks ---
    def oe_merge_sort(n):
        """Odd-even merge sort network for n elements."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = oe_merge_sort(half)
        right = oe_merge_sort(n - half)
        right = [(i + half, j + half) for i, j in right]
        
        # Odd-even merge of two sorted halves
        merge = []
        for phase in range(n - 1):
            offset = phase % 2
            for i in range(offset, n - 1, 2):
                merge.append((i, i + 1))
        
        return left + right + merge

    def odd_even_transposition(n):
        """Odd-even transposition network for n elements."""
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps

    # Build multiple diverse seed networks
    seed1 = oe_merge_sort(N)
    seed2 = odd_even_transposition(N)
    
    # Verify seeds
    if not is_sorted_network_full(from_layers(to_layers(seed1))):
        seed1 = odd_even_transposition(N)
    if not is_sorted_network_full(from_layers(to_layers(seed2))):
        seed2 = oe_merge_sort(N)
    
    # Build a third seed: a "random" valid network via greedy construction
    def greedy_sort_network():
        """Build a sorting network greedily: at each step, add a comparator
        that reduces the number of unsorted sample inputs the most."""
        comps = []
        # Start with all 2^17 inputs, track which are sorted
        # For efficiency, use sample
        sample = _SAMPLE_INPUTS
        expected = _SAMPLE_SORTED
        state = sample.copy()
        sorted_mask = np.array_equal(state, expected, axis=1)
        n_unsorted = (~sorted_mask).sum()
        
        # Precompute all possible comparators
        all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
        rng_g = random.Random(999)
        rng_g.shuffle(all_comps)
        
        max_iter = 200
        for _ in range(max_iter):
            if n_unsorted == 0:
                break
            best_comp = None
            best_unsorted = n_unsorted
            # Try a random subset of comparators
            for comp in all_comps[:50]:
                i, j = comp
                a = state[:, i]
                b = state[:, j]
                mask = a > b
                new_i = np.where(mask, b, a)
                new_j = np.where(mask, a, b)
                # Only check unsorted rows for speed
                unsorted_idx = np.where(~sorted_mask)[0]
                if len(unsorted_idx) == 0:
                    break
                temp_state = state.copy()
                temp_state[unsorted_idx, i] = new_i[unsorted_idx]
                temp_state[unsorted_idx, j] = new_j[unsorted_idx]
                still_unsorted = (~np.array_equal(temp_state[unsorted_idx], expected[unsorted_idx], axis=1)).sum()
                if still_unsorted < best_unsorted:
                    best_unsorted = still_unsorted
                    best_comp = comp
            if best_comp is None:
                break
            i, j = best_comp
            a = state[:, i]
            b = state[:, j]
            mask = a > b
            state[:, i] = np.where(mask, b, a)
            state[:, j] = np.where(mask, a, b)
            sorted_mask = np.array_equal(state, expected, axis=1)
            n_unsorted = (~sorted_mask).sum()
            comps.append(best_comp)
        
        return comps

    seed3 = greedy_sort_network()
    if not is_sorted_network_full(from_layers(to_layers(seed3))):
        seed3 = seed1  # fallback

    # Convert seeds to layers
    pop = []
    for seed in [seed1, seed2, seed3]:
        layers = to_layers(seed)
        if is_valid_network(layers):
            pop.append(layers)
    
    # If we don't have enough valid seeds, add more
    while len(pop) < 3:
        layers = to_layers(seed1)
        pop.append(layers)

    # --- Genetic Algorithm with layer-based representation ---
    rng_ga = random.Random(777)
    best_layers = None
    best_count = float('inf')
    
    # Track best
    for ind in pop:
        c = count_comparators(ind)
        if c < best_count:
            best_count = c
            best_layers = [list(layer) for layer in ind]

    def mutate_layers(layers):
        """Apply random mutations to a layer-based network."""
        layers = [list(layer) for layer in layers]
        num_mutations = rng_ga.choice([1, 1, 2, 2, 3])
        
        for _ in range(num_mutations):
            op = rng_ga.random()
            
            if op < 0.4 and len(layers) > 0:
                # Remove a random comparator from a random layer
                layer_idx = rng_ga.randint(0, len(layers) - 1)
                if len(layers[layer_idx]) > 0:
                    comp_idx = rng_ga.randint(0, len(layers[layer_idx]) - 1)
                    layers[layer_idx].pop(comp_idx)
                    if len(layers[layer_idx]) == 0:
                        layers.pop(layer_idx)
                        
            elif op < 0.7 and len(layers) > 0:
                # Add a random comparator to a random layer (if it doesn't conflict)
                layer_idx = rng_ga.randint(0, len(layers) - 1)
                i = rng_ga.randint(0, N - 2)
                j = rng_ga.randint(i + 1, N - 1)
                # Check conflict
                used = set()
                for ci, cj in layers[layer_idx]:
                    used.add(ci)
                    used.add(cj)
                if i not in used and j not in used:
                    layers[layer_idx].append((i, j))
                    
            elif op < 0.85 and len(layers) > 1:
                # Swap two adjacent layers
                layer_idx = rng_ga.randint(0, len(layers) - 2)
                layers[layer_idx], layers[layer_idx + 1] = layers[layer_idx + 1], layers[layer_idx]
                
            else:
                # Move a comparator to a different layer
                if len(layers) > 1:
                    src_layer = rng_ga.randint(0, len(layers) - 1)
                    if len(layers[src_layer]) > 0:
                        comp_idx = rng_ga.randint(0, len(layers[src_layer]) - 1)
                        comp = layers[src_layer].pop(comp_idx)
                        if len(layers[src_layer]) == 0:
                            layers.pop(src_layer)
                        # Find a layer where it fits
                        dst_layer = rng_ga.randint(0, len(layers) - 1)
                        used = set()
                        for ci, cj in layers[dst_layer]:
                            used.add(ci)
                            used.add(cj)
                        if comp[0] not in used and comp[1] not in used:
                            layers[dst_layer].append(comp)
                        else:
                            # Put it back
                            if src_layer < len(layers):
                                layers[src_layer].append(comp)
                            else:
                                layers.append([comp])
        
        return layers

    def crossover_layers(parent1, parent2):
        """Crossover: combine layers from two parents."""
        max_depth = max(len(parent1), len(parent2))
        child = []
        for d in range(max_depth):
            if d < len(parent1) and d < len(parent2):
                # Choose from either parent's layer at this depth
                if rng_ga.random() < 0.5:
                    child.append(list(parent1[d]))
                else:
                    child.append(list(parent2[d]))
            elif d < len(parent1):
                child.append(list(parent1[d]))
            else:
                child.append(list(parent2[d]))
        # Remove empty layers
        child = [layer for layer in child if len(layer) > 0]
        return child

    def fitness(layers):
        """Graded fitness: penalize invalidity and size."""
        c = count_comparators(layers)
        if c > 4096:
            return -1e6
        # Quick sample check
        state = simulate_layers(layers, _SAMPLE_INPUTS)
        mismatches = (~np.array_equal(state, _SAMPLE_SORTED, axis=1)).sum()
        # Fitness: lower is better. Penalize mismatches heavily, then size.
        return c + mismatches * 100

    POP_SIZE = 8
    ELITE_SIZE = 2
    MAX_GENERATIONS = 200
    
    # Initialize population with diversity
    for _ in range(POP_SIZE - len(pop)):
        base = pop[rng_ga.randint(0, len(pop) - 1)]
        mutant = mutate_layers(base)
        if count_comparators(mutant) <= 4096:
            pop.append(mutant)
    
    generation = 0
    no_improve_count = 0
    
    while time.time() < deadline - 3.0 and generation < MAX_GENERATIONS:
        generation += 1
        
        # Evaluate fitness
        scored = [(fitness(ind), ind) for ind in pop]
        scored.sort(key=lambda x: x[0])
        
        # Update best
        best_score, best_ind = scored[0]
        best_c = count_comparators(best_ind)
        if best_c < best_count:
            # Verify with full check
            if is_valid_network(best_ind):
                best_count = best_c
                best_layers = [list(layer) for layer in best_ind]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Early termination if we hit the theoretical lower bound
        if best_count <= 63:
            break
        
        # Selection: keep elites
        new_pop = []
        for i in range(ELITE_SIZE):
            new_pop.append([list(layer) for layer in scored[i][1]])
        
        # Generate offspring
        while len(new_pop) < POP_SIZE:
            if time.time() > deadline - 3.0:
                break
            
            # Tournament selection for parents
            p1_idx = rng_ga.randint(0, min(4, len(scored) - 1))
            p2_idx = rng_ga.randint(0, min(4, len(scored) - 1))
            p1 = scored[p1_idx][1]
            p2 = scored[p2_idx][1]
            
            r = rng_ga.random()
            if r < 0.5:
                child = crossover_layers(p1, p2)
            else:
                child = mutate_layers(p1)
            
            if count_comparators(child) <= 4096:
                new_pop.append(child)
        
        pop = new_pop
        
        # Occasionally inject a mutated seed to maintain diversity
        if generation % 10 == 0:
            seed_choice = rng_ga.choice([seed1, seed2, seed3])
            mutant = mutate_layers(to_layers(seed_choice))
            if count_comparators(mutant) <= 4096:
                pop[-1] = mutant
    
    # --- Final local search on best solution: aggressive deletion ---
    if best_layers is not None and time.time() < deadline - 1.0:
        current = [list(layer) for layer in best_layers]
        
        def try_remove_from_layers(layers, layer_idx, comp_idx):
            candidate = [list(layer) for layer in layers]
            candidate[layer_idx].pop(comp_idx)
            if len(candidate[layer_idx]) == 0:
                candidate.pop(layer_idx)
            if is_valid_sample(candidate):
                if is_valid_network(candidate):
                    return True, candidate
            return False, layers
        
        # Multiple deletion passes
        for _pass in range(30):
            if time.time() > deadline - 1.0:
                break
            progress = False
            for li in range(len(current)):
                if time.time() > deadline - 1.0:
                    break
                ci = 0
                while ci < len(current[li]):
                    if time.time() > deadline - 1.0:
                        break
                    ok, new_net = try_remove_from_layers(current, li, ci)
                    if ok:
                        current = new_net
                        progress = True
                        # Don't increment ci
                    else:
                        ci += 1
            if not progress:
                break
        
        # Update best
        c = count_comparators(current)
        if c < best_count:
            if is_valid_network(current):
                best_count = c
                best_layers = [list(layer) for layer in current]

    # --- Return as flat list ---
    if best_layers is None:
        # Fallback: return seed
        return seed1
    
    result = from_layers(best_layers)
    
    # Final validation
    if not is_sorted_network_full(result):
        return seed1
    
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
