"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import itertools


def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = chosen[i], chosen[j]
            d_pa = _dist_sq(p, a)
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False


def _is_maximal(chosen):
    """Check if chosen is maximal: no other point can be added without creating isosceles."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p not in chosen_set:
            if not _creates_isosceles(p, chosen):
                return False
    return True


def _make_maximal(chosen):
    """Greedily expand chosen to be maximal by adding non-conflicting points."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p not in chosen_set:
            if not _creates_isosceles(p, chosen):
                chosen.append(p)
                chosen_set.add(p)
    return chosen


def _random_valid_set():
    """Generate a random valid no-isosceles set (not necessarily maximal)."""
    chosen = []
    all_points = list(itertools.product(range(N), repeat=2))
    random.shuffle(all_points)
    for p in all_points:
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
    return chosen


def _try_remove(chosen, idx):
    """Try removing point at idx and check if result can still be made maximal."""
    if len(chosen) <= 1:
        return None
    new_chosen = chosen[:idx] + chosen[idx+1:]
    # Check if new_chosen is still a valid no-isosceles set
    # It is by construction since we're removing a point
    # Now check if it can be extended to a maximal set
    extended = _make_maximal(new_chosen)
    if len(extended) < len(chosen):
        return extended
    return None


def _try_swap(chosen, idx, new_p):
    """Try replacing point at idx with new_p."""
    if new_p in chosen:
        return None
    new_chosen = [p for i, p in enumerate(chosen) if i != idx]
    if _creates_isosceles(new_p, new_chosen):
        return None
    new_chosen.append(new_p)
    extended = _make_maximal(new_chosen)
    if len(extended) < len(chosen):
        return extended
    return None


def construct():
    deadline = time.time() + 280  # Safety margin before 300s limit
    best_set = None
    best_size = float('inf')
    
    # Initial solution: greedy row-major
    chosen = []
    for p in itertools.product(range(N), repeat=2):
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
    best_set = list(chosen)
    best_size = len(chosen)
    
    # Try multiple random restarts with local search
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start with a random valid set and make it maximal
        current = _make_maximal(_random_valid_set())
        current_size = len(current)
        
        if current_size < best_size:
            best_set = list(current)
            best_size = current_size
        
        # Local search with simulated annealing
        temperature = 1.0
        for step in range(2000):
            if time.time() > deadline:
                break
            
            if current_size < best_size:
                best_set = list(current)
                best_size = current_size
            
            # Try random removal
            if random.random() < 0.5:
                idx = random.randint(0, len(current) - 1)
                new_set = _try_remove(current, idx)
                if new_set is not None:
                    current = new_set
                    current_size = len(current)
                else:
                    # Simulated annealing: accept worse with some probability
                    if temperature > 0.01 and random.random() < temperature:
                        # Try a different removal
                        idx2 = random.randint(0, len(current) - 1)
                        if idx2 != idx:
                            new_set2 = _try_remove(current, idx2)
                            if new_set2 is not None:
                                current = new_set2
                                current_size = len(current)
            else:
                # Try random swap
                idx = random.randint(0, len(current) - 1)
                new_p = tuple(random.randint(0, N-1) for _ in range(2))
                new_set = _try_swap(current, idx, new_p)
                if new_set is not None:
                    current = new_set
                    current_size = len(current)
            
            # Cool down
            temperature *= 0.995
        
        # Occasionally try a more aggressive restart
        if restart % 5 == 4 and time.time() < deadline:
            # Try starting from a smaller random subset
            all_points = list(itertools.product(range(N), repeat=2))
            random.shuffle(all_points)
            small_start = all_points[:random.randint(3, 8)]
            # Filter to valid no-isosceles
            valid_start = []
            for p in small_start:
                if not _creates_isosceles(p, valid_start):
                    valid_start.append(p)
            current = _make_maximal(valid_start)
            current_size = len(current)
            if current_size < best_size:
                best_set = list(current)
                best_size = current_size
    
    return best_set
# EVOLVE-BLOCK-END
