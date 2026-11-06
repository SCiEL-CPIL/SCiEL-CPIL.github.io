import time
import numpy as np
import random

N = 18
TIME_LIMIT = 55

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Pre-compute bit-parallel states for fast verification
    # 2^18 inputs.
    # We pack 64 inputs per uint64.
    num_blocks = 2**N // 64
    
    # states[w][block] bit b is 1 if input (block*64+b) has 1 on wire w
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Fill states using vectorized approach per wire
    for w in range(N):
        for b in range(num_blocks):
            base = b * 64
            if w < 6:
                # Generate the 64-bit mask for this block
                mask = 0
                for bit in range(64):
                    if ((base + bit) >> w) & 1:
                        mask |= (1 << bit)
                states[w][b] = mask
            else:
                # w >= 6: all 64 numbers in a block have the same w-th bit
                if (base >> w) & 1:
                    states[w][b] = (1 << 64) - 1
                else:
                    states[w][b] = 0

    # Pre-allocate workspace for verification
    # We use a list of arrays. To avoid re-allocation, we copy states into workspace.
    workspace = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    def verify(comparators):
        """Verify network sorts all inputs."""
        # Copy initial states into workspace
        for w in range(N):
            workspace[w][:] = states[w]
        
        # Apply comparators
        for i, j in comparators:
            a = workspace[i]
            b = workspace[j]
            # Use temporary variables to avoid intermediate array creation if possible?
            # Numpy & and | create new arrays. 
            # To optimize, we could use bitwise ops on a single large array, but 
            # given the constraints, this is the standard fast path.
            workspace[i] = a & b
            workspace[j] = a | b
        
        # Check sorted
        for w in range(N - 1):
            if np.any(workspace[w] & ~workspace[w + 1]):
                return False
        return True

    def greedy_reduce(net):
        """Remove redundant comparators using fast verification."""
        cur_net = list(net)
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            idx = 0
            while idx < len(cur_net):
                if time.time() > deadline - 2:
                    return cur_net
                # Try removing comparator at idx
                trial = cur_net[:idx] + cur_net[idx+1:]
                if verify(trial):
                    cur_net = trial
                    changed = True
                else:
                    idx += 1
        return cur_net

    def random_layer():
        """Generate a random valid layer (set of disjoint comparators)."""
        wires = list(range(N))
        random.shuffle(wires)
        layer = []
        i = 0
        while i < N - 1:
            if random.random() < 0.6: # Probability of placing a comparator
                a, b = wires[i], wires[i+1]
                if a > b:
                    a, b = b, a
                layer.append((a, b))
                i += 2
            else:
                i += 1
        return layer

    def to_layers(net):
        """Convert a flat list of comparators to a list of layers."""
        layers = []
        remaining = list(net)
        while remaining:
            layer = []
            used_wires = set()
            new_remaining = []
            for comp in remaining:
                i, j = comp
                if i not in used_wires and j not in used_wires:
                    layer.append(comp)
                    used_wires.add(i)
                    used_wires.add(j)
                else:
                    new_remaining.append(comp)
            if not layer:
                # Should not happen if net is valid, but just in case
                layer = [remaining[0]]
                new_remaining = remaining[1:]
            layers.append(layer)
            remaining = new_remaining
        return layers

    def from_layers(layers):
        """Flatten layers into a list of comparators."""
        net = []
        for layer in layers:
            net.extend(layer)
        return net

    # Initial Network: Start with a known good structure.
    # Seed is 153. Reduce it.
    best_network = _odd_even_transposition()
    best_network = greedy_reduce(best_network)
    best_len = len(best_network)
    
    # Convert to layers for the new search representation
    current_layers = to_layers(best_network)
    
    stagnation_counter = 0
    max_stagnation = 100
    
    while time.time() < deadline - 5:
        # Choose a move on the layer structure
        move_type = random.random()
        
        if move_type < 0.3:
            # Mutation: Replace a random layer with a new random layer
            if current_layers:
                idx = random.randint(0, len(current_layers) - 1)
                current_layers[idx] = random_layer()
                
        elif move_type < 0.5:
            # Insertion: Insert a new random layer
            idx = random.randint(0, len(current_layers))
            current_layers.insert(idx, random_layer())
            
        elif move_type < 0.7:
            # Deletion: Try to remove a layer
            if len(current_layers) > 1:
                idx = random.randint(0, len(current_layers) - 1)
                current_layers.pop(idx)
                
        elif move_type < 0.85:
            # Swap: Swap two layers
            if len(current_layers) > 1:
                i, j = random.sample(range(len(current_layers)), 2)
                current_layers[i], current_layers[j] = current_layers[j], current_layers[i]
        else:
            # Perturb Layer: Change one comparator in a random layer
            if current_layers:
                l_idx = random.randint(0, len(current_layers) - 1)
                layer = current_layers[l_idx]
                if layer:
                    c_idx = random.randint(0, len(layer) - 1)
                    # Generate a new comparator that doesn't conflict with others in the layer
                    other_wires = set()
                    for k, (a, b) in enumerate(layer):
                        if k != c_idx:
                            other_wires.add(a)
                            other_wires.add(b)
                    
                    # Try to find a new valid comparator
                    new_comp = None
                    for _ in range(10):
                        a, b = random.sample(range(N), 2)
                        if a > b: a, b = b, a
                        if a not in other_wires and b not in other_wires:
                            new_comp = (a, b)
                            break
                    if new_comp:
                        layer[c_idx] = new_comp
                    else:
                        # If we can't find a valid one, remove the comparator
                        layer.pop(c_idx)
                        if not layer:
                            current_layers.pop(l_idx)

        # Flatten and reduce
        trial_net = from_layers(current_layers)
        reduced = greedy_reduce(trial_net)
        
        # Update best
        if len(reduced) < best_len:
            best_network = reduced
            best_len = len(reduced)
            current_layers = to_layers(reduced)
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
            if stagnation_counter >= max_stagnation:
                # Kick: Perturb the best network's layers significantly
                K = random.randint(2, 5)
                for _ in range(K):
                    if not current_layers: break
                    idx = random.randint(0, len(current_layers) - 1)
                    current_layers[idx] = random_layer()
                
                current_layers = to_layers(greedy_reduce(from_layers(current_layers)))
                stagnation_counter = 0

    return best_network
