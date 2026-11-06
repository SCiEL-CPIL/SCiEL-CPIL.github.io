"""Layer-based genetic algorithm for size-efficient sorting networks on 17 wires."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136
_NUM_INPUTS = 1 << N

# Precompute all Boolean inputs once (uint8 for fast bitwise ops)
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)[:, None]
_bits = np.arange(N, dtype=np.uint32)[None, :]
_ALL_INPUTS = (( _indices >> _bits) & 1).astype(np.uint8)

# Precomputed per-comparator min/max results on the full input set.
# For comparator (i, j): wire i gets AND, wire j gets OR (for 0/1 values).
# We precompute these column-pairs so validation is just array copies.
def _precompute_cmp():
    # cmp_and[i, j] = _ALL_INPUTS[:,i] & _ALL_INPUTS[:,j]
    # cmp_or[i, j]  = _ALL_INPUTS[:,i] | _ALL_INPUTS[:,j]
    cmp_and = np.empty((N, N), dtype=object)
    cmp_or = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            cmp_and[i, j] = _ALL_INPUTS[:, i] & _ALL_INPUTS[:, j]
            cmp_or[i, j] = _ALL_INPUTS[:, i] | _ALL_INPUTS[:, j]
    return cmp_and, cmp_or

_CMP_AND, _CMP_OR = _precompute_cmp()

def _simulate(comparators):
    """Simulate network on all inputs using precomputed columns. Returns output array."""
    state = _ALL_INPUTS.copy()
    for i, j in comparators:
        state[:, i] = _CMP_AND[i, j]
        state[:, j] = _CMP_OR[i, j]
    return state

def is_sorted_network(comparators):
    """Check if the network sorts all 2^17 inputs (zero-one principle)."""
    if not comparators or len(comparators) > 4096:
        return False
    out = _simulate(comparators)
    # sorted iff no adjacent inversion: out[:,k] <= out[:,k+1] for all k
    for k in range(N - 1):
        if np.any(out[:, k] > out[:, k + 1]):
            return False
    return True

# ---- Layer representation helpers ----
# A layer is a list of (i, j) comparators where no wire is used twice (a matching).
# A network is a list of layers. Flattened = layers concatenated in order.

def _layers_to_flat(layers):
    flat = []
    for layer in layers:
        flat.extend(layer)
    return flat

def _flat_to_layers(flat):
    """Group a flat comparator list into parallel layers (greedy by wire availability)."""
    layers = []
    used = [False] * N
    cur = []
    for c in flat:
        i, j = c
        if used[i] or used[j]:
            layers.append(cur)
            used = [False] * N
            cur = []
        used[i] = True
        used[j] = True
        cur.append(c)
    if cur:
        layers.append(cur)
    return layers

def _is_valid_layer(layer):
    """A layer must be a matching: no wire repeated, and i < j."""
    seen = [False] * N
    for i, j in layer:
        if i >= j:
            return False
        if seen[i] or seen[j]:
            return False
        seen[i] = True
        seen[j] = True
    return True

# ---- Seed network (odd-even transposition) as layers ----
def _seed_network():
    seed = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed.append((i, i + 1))
    return seed

def _seed_layers():
    # Odd-even transposition is already structured in parallel layers.
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        if layer:
            layers.append(layer)
    return layers

# ---- GA operators ----
def _random_layer():
    """Generate a random matching layer (possibly empty)."""
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    used = [False] * N
    for w in wires:
        if len(layer) >= N // 2:
            break
        # pick a partner not yet used
        options = [v for v in range(N) if v != w and not used[w] and not used[v]]
        if not options:
            continue
        v = random.choice(options)
        a, b = (w, v) if w < v else (v, w)
        used[a] = True
        used[b] = True
        layer.append((a, b))
    return layer

def _crossover_layers(A, B):
    """Uniform crossover over layers: for each layer position, pick from A or B."""
    max_len = max(len(A), len(B))
    child = []
    for k in range(max_len):
        if random.random() < 0.5:
            if k < len(A):
                child.append(list(A[k]))
            else:
                child.append([])
        else:
            if k < len(B):
                child.append(list(B[k]))
            else:
                child.append([])
    # drop empty layers
    return [L for L in child if L]

def _mutate_layers(layers, p=0.15):
    """Mutation: randomly add/remove/swap comparators within layers."""
    new_layers = [list(L) for L in layers]
    # Add a random layer
    if random.random() < p:
        pos = random.randint(0, len(new_layers))
        new_layers.insert(pos, _random_layer())
    # Remove a random layer
    if new_layers and random.random() < p:
        pos = random.randint(0, len(new_layers) - 1)
        new_layers.pop(pos)
    # Mutate a random layer: add/remove/swap a comparator
    if new_layers and random.random() < p:
        li = random.randint(0, len(new_layers) - 1)
        layer = new_layers[li]
        choice = random.random()
        if choice < 0.33:
            # add a comparator that doesn't conflict
            free = [w for w in range(N) if w not in layer]
            if len(free) >= 2:
                a, b = random.sample(free, 2)
                if a > b:
                    a, b = b, a
                layer.append((a, b))
        elif choice < 0.66:
            # remove a comparator
            if layer:
                layer.pop(random.randint(0, len(layer) - 1))
        else:
            # swap a comparator with a random one
            if layer:
                idx = random.randint(0, len(layer) - 1)
                used = [False] * N
                for (x, y) in layer:
                    if (x, y) != (layer[idx][0], layer[idx][1]):
                        used[x] = True
                        used[y] = True
                free = [w for w in range(N) if not used[w]]
                if len(free) >= 2:
                    a, b = random.sample(free, 2)
                    if a > b:
                        a, b = b, a
                    layer[idx] = (a, b)
    # drop empty layers
    return [L for L in new_layers if L]

def _fitness(comparators):
    """Lower is better: comparator count + penalty for unsortedness."""
    if not comparators:
        return float('inf')
    out = _simulate(comparators)
    # count inversions (number of rows not sorted)
    bad = 0
    for k in range(N - 1):
        bad += int(np.sum(out[:, k] > out[:, k + 1]))
    return len(comparators) + 0.01 * bad

# ---- Local deletion refinement ----
def _local_delete(flat, deadline):
    """Systematically try to remove each comparator."""
    current = list(flat)
    i = 0
    while i < len(current):
        if time.time() > deadline:
            break
        candidate = current[:i] + current[i + 1:]
        if is_sorted_network(candidate):
            current = candidate
        else:
            i += 1
    return current

def _local_delete_multi(flat, max_passes, deadline):
    current = list(flat)
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        before = len(current)
        current = _local_delete(current, deadline)
        if len(current) == before:
            break
    return current

# ---- Main construct ----
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill

    seed = _seed_network()
    best = list(seed)
    best_len = len(seed)

    # Phase 1: GA over layer representations
    population = []
    # seed the population with the seed network and some random layer networks
    seed_layers = _seed_layers()
    population.append(seed_layers)
    for _ in range(15):
        # random network: random number of layers (depth 8-16)
        depth = random.randint(8, 16)
        layers = []
        for _ in range(depth):
            layers.append(_random_layer())
        population.append([L for L in layers if L])

    def _pop_fitness(layers):
        return _fitness(_layers_to_flat(layers))

    # evaluate initial population
    pop_fit = []
    for ind in population:
        flat = _layers_to_flat(ind)
        if is_sorted_network(flat):
            pop_fit.append(len(flat))  # valid: fitness = size
        else:
            pop_fit.append(_fitness(flat))

    # GA loop
    ga_deadline = deadline - 20.0
    best_layers = [list(L) for L in seed_layers]
    best_flat = list(seed)

    generation = 0
    while time.time() < ga_deadline and generation < 60:
        generation += 1
        # sort population by fitness
        order = sorted(range(len(population)), key=lambda k: pop_fit[k])
        # elitism: keep best 4
        survivors = [population[k] for k in order[:4]]
        # if best valid so far, record
        for k in order:
            flat = _layers_to_flat(population[k])
            if is_sorted_network(flat) and len(flat) < best_len:
                best = flat
                best_len = len(flat)
                best_layers = [list(L) for L in population[k]]

        # build next generation
        new_pop = [list(L) for L in survivors]
        # tournament selection + crossover + mutation
        while len(new_pop) < len(population):
            # tournament of 3
            cands = random.sample(range(len(population)), 3)
            cands.sort(key=lambda k: pop_fit[k])
            pa = population[cands[0]]
            pb = population[random.choice(cands)]
            child = _crossover_layers(pa, pb)
            if random.random() < 0.6:
                child = _mutate_layers(child, p=0.2)
            if child:
                new_pop.append(child)

        population = new_pop
        # re-evaluate
        pop_fit = []
        for ind in population:
            flat = _layers_to_flat(ind)
            if is_sorted_network(flat):
                pop_fit.append(len(flat))
            else:
                pop_fit.append(_fitness(flat))

    # Use best found by GA
    if best_len < len(seed):
        best = _local_delete_multi(best, max_passes=15, deadline=deadline - 8.0)
        best_len = len(best)

    # Phase 2: aggressive local deletion from best
    best = _local_delete_multi(best, max_passes=25, deadline=deadline - 5.0)
    best_len = len(best)

    # Phase 3: random restart local search if time remains
    if time.time() < deadline - 4.0:
        for _ in range(10):
            if time.time() > deadline - 4.0:
                break
            cand = list(best)
            for _ in range(30):
                if time.time() > deadline - 4.0:
                    break
                if not cand:
                    break
                idx = random.randint(0, len(cand) - 1)
                nc = cand[:idx] + cand[idx + 1:]
                if is_sorted_network(nc):
                    cand = nc
            if len(cand) < best_len:
                best = cand
                best_len = len(cand)
                best = _local_delete_multi(best, max_passes=10, deadline=deadline - 4.0)
                best_len = len(best)

    # Final validation
    if not is_sorted_network(best):
        return seed
    return best
