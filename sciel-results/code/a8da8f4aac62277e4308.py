"""Search-based constructor for size-efficient sorting networks on 18 wires."""
import time
import random
import sys

N = 18
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _apply_network(net, state):
    """Apply comparator network to a state (list of 0/1). Returns sorted state if correct."""
    for i, j in net:
        if state[i] > state[j]:
            state[i], state[j] = state[j], state[i]
    return state

def _is_sorted_net(net):
    """Check if network sorts all 2^18 boolean inputs."""
    for mask in range(1 << N):
        state = [(mask >> i) & 1 for i in range(N)]
        for i, j in net:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        zeros = 0
        for v in state:
            if v == 0:
                zeros += 1
        for k in range(zeros):
            if state[k] != 0:
                return False
        for k in range(zeros, N):
            if state[k] != 1:
                return False
    return True

def _is_sorted_net_fast(net):
    """Faster check using bitmask representation."""
    for mask in range(1 << N):
        state = mask
        for i, j in net:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi > bj:
                state = (state & ~(1 << i)) | (1 << j)
        ones = bin(state).count('1')
        expected = ((1 << ones) - 1) << (N - ones)
        if state != expected:
            return False
    return True

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _bubble_sort_network():
    """Generate a bubble sort network."""
    net = []
    for _ in range(N):
        for i in range(N - 1):
            net.append((i, i + 1))
    return net

def _random_network(num_comparators, rng):
    """Generate a random network with a given number of comparators."""
    net = []
    for _ in range(num_comparators):
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        net.append((i, j))
    return net

def _improve(net, deadline):
    """Try to improve a network by removing redundant comparators."""
    current = list(net)
    best = list(current)
    best_len = len(current)
    
    # First pass: try removing each comparator
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = current[:idx] + current[idx+1:]
            if _is_sorted_net_fast(candidate):
                current = candidate
                best = list(current)
                best_len = len(current)
                changed = True
    
    # Second pass: try swapping adjacent comparators
    for _ in range(100):
        if time.time() > deadline:
            break
        if not current:
            break
        rng = random.Random()
        i = rng.randint(0, len(current) - 2)
        current[i], current[i+1] = current[i+1], current[i]
        if _is_sorted_net_fast(current):
            if len(current) < best_len:
                best = list(current)
                best_len = len(current)
        else:
            current[i], current[i+1] = current[i+1], current[i]
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with the seed network
    seed_net = _odd_even_transposition()
    best_net = list(seed_net)
    best_len = len(seed_net)
    
    # Verify seed
    if not _is_sorted_net_fast(seed_net):
        raise ValueError("Seed network is invalid")
    
    # Try to improve the seed
    if time.time() < deadline - 5:
        improved = _improve(seed_net, deadline - 5)
        if len(improved) < best_len:
            best_net = improved
            best_len = len(improved)
    
    # Try bubble sort network
    if time.time() < deadline - 5:
        bubble_net = _bubble_sort_network()
        if _is_sorted_net_fast(bubble_net):
            if len(bubble_net) < best_len:
                best_net = list(bubble_net)
                best_len = len(bubble_net)
            improved = _improve(bubble_net, deadline - 5)
            if len(improved) < best_len:
                best_net = improved
                best_len = len(improved)
    
    # Try random networks of various sizes
    rng = random.Random(12345)
    for size in [100, 110, 120, 130, 140]:
        if time.time() > deadline - 5:
            break
        rand_net = _random_network(size, rng)
        if _is_sorted_net_fast(rand_net):
            if len(rand_net) < best_len:
                best_net = list(rand_net)
                best_len = len(rand_net)
            improved = _improve(rand_net, deadline - 5)
            if len(improved) < best_len:
                best_net = improved
                best_len = len(improved)
    
    # Final verification
    if not _is_sorted_net_fast(best_net):
        best_net = seed_net
    
    if len(best_net) > 4096:
        best_net = best_net[:4096]
    
    return best_net

# EVOLVE-BLOCK-END
