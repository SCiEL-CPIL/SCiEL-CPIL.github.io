"""Combined program: layer-based representation with greedy removal + simulated annealing."""
import time
import random
import numpy as np

N = 18
DEADLINE = 55.0  # safety margin before 60s kill

# Precompute all 2^N boolean inputs
all_masks = np.arange(1 << N, dtype=np.int32)
W0_full = np.zeros((1 << N, N), dtype=np.int8)
for i in range(N):
    W0_full[:, i] = (all_masks >> i) & 1

# Subset for fast rejection: first 2048 + 2048 random
_subset_rng = np.random.RandomState(42)
subset_indices = np.concatenate([
    np.arange(2048),
    _subset_rng.choice(1 << N, 2048, replace=False)
])
W0_subset = W0_full[subset_indices]


def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses the 0-1 principle with vectorized bitwise operations.
    Returns True if valid.
    """
    if len(net) > 4096:
        return False
    if len(net) == 0:
        return N <= 1

    # Fast subset check first
    W_sub = W0_subset.copy()
    for i, j in net:
        col_i = W_sub[:, i]
        col_j = W_sub[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp

    if np.any(W_sub[:, :-1] > W_sub[:, 1:]):
        return False

    # Full check
    W_full = W0_full.copy()
    for i, j in net:
        col_i = W_full[:, i]
        col_j = W_full[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp

    if np.any(W_full[:, :-1] > W_full[:, 1:]):
        return False

    return True


def _odd_even_transposition():
    """Generate the odd-even transposition sorting network as a flat list."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _flat_to_layers(net):
    """Convert flat comparator list to list of layers (each layer is a set of disjoint comparators)."""
    # A layer is a set of comparators that can be executed in parallel (no shared wires)
    layers = []
    used = [False] * N
    for comp in net:
        i, j = comp
        # Find or create a layer where both i and j are free
        placed = False
        for layer in layers:
            if not used[i] and not used[j]:
                layer.append(comp)
                used[i] = True
                used[j] = True
                placed = True
                break
        if not placed:
            # Try to find a layer where we can fit it by checking actual disjointness
            for li, layer in enumerate(layers):
                wires_in_layer = set()
                for c in layer:
                    wires_in_layer.add(c[0])
                    wires_in_layer.add(c[1])
                if i not in wires_in_layer and j not in wires_in_layer:
                    layer.append(comp)
                    placed = True
                    break
        if not placed:
            layers.append([comp])
    return layers


def _layers_to_flat(layers):
    """Convert list of layers to flat comparator list."""
    net = []
    for layer in layers:
        net.extend(layer)
    return net


def _random_layer():
    """Generate a random layer: a random matching on a random subset of wires."""
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    for i in range(0, len(wires) - 1, 2):
        a, b = wires[i], wires[i + 1]
        if a > b:
            a, b = b, a
        layer.append((a, b))
    return layer


def _mutate_layer(layer):
    """Mutate a single layer: change some comparators in the matching."""
    new_layer = []
    wires = list(range(N))
    random.shuffle(wires)
    # Keep some fraction of original structure
    for comp in layer:
        if random.random() < 0.7:
            # Keep this comparator
            new_layer.append(comp)
            wires.remove(comp[0])
            wires.remove(comp[1])
        # else: drop it, wires remain available
    # Fill remaining with random disjoint comparators from available wires
    random.shuffle(wires)
    for i in range(0, len(wires) - 1, 2):
        a, b = wires[i], wires[i + 1]
        if a > b:
            a, b = b, a
        # Check disjointness with existing
        existing_wires = set()
        for c in new_layer:
            existing_wires.add(c[0])
            existing_wires.add(c[1])
        if a not in existing_wires and b not in existing_wires:
            new_layer.append((a, b))
    return new_layer


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Start with odd-even transposition
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)

    # Phase 1: Greedy removal (try removing each comparator, keep if still valid)
    changed = True
    while changed:
        if time.time() > deadline - 10:
            break
        changed = False

        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            new_net = net[:idx] + net[idx + 1:]
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break

        if time.time() > deadline - 10:
            break

        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 10:
                    break
                new_net = net[:idx] + net[idx + 1:]
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break

    # Phase 2: Layer-based simulated annealing
    if time.time() < deadline - 5:
        T = 1.0
        min_T = 0.05
        cooling = 0.999
        current_count = len(net)
        layers = _flat_to_layers(net)

        while time.time() < deadline - 3:
            if T < min_T:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                layers = _flat_to_layers(net)
                T = 1.0

            r = random.random()
            new_net = None

            if r < 0.3 and len(net) > 0:
                # Remove a random comparator
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx + 1:]
            elif r < 0.5:
                # Insert a random comparator at a random position
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            elif r < 0.7:
                # Mutate a random comparator
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:]
                new_net[idx] = (i, j)
            elif r < 0.85:
                # Layer-based mutation: mutate one layer
                if len(layers) == 0:
                    continue
                li = random.randint(0, len(layers) - 1)
                new_layers = layers[:]
                new_layers[li] = _mutate_layer(layers[li])
                new_net = _layers_to_flat(new_layers)
            else:
                # Swap two layers
                if len(layers) < 2:
                    continue
                li, lj = random.sample(range(len(layers)), 2)
                new_layers = layers[:]
                new_layers[li], new_layers[lj] = new_layers[lj], new_layers[li]
                new_net = _layers_to_flat(new_layers)

            if new_net is None or len(new_net) > 4096:
                continue

            if not _is_sorted_network(new_net):
                continue

            new_count = len(new_net)
            delta = new_count - current_count

            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                layers = _flat_to_layers(net)
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count

            T *= cooling

    # Final validation
    if not _is_sorted_network(best_net):
        best_net = _odd_even_transposition()

    return best_net
