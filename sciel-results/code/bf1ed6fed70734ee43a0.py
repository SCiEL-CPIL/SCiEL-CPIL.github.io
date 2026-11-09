import time
import random

N = 17
TOTAL_MASKS = 1 << N  # 131072

# EVOLVE-BLOCK-START

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _init_wire_ints():
    """Initialize 17 wire integers where bit m represents mask m's value on that wire."""
    wires = []
    for i in range(N):
        w = 0
        for m in range(TOTAL_MASKS):
            if (m >> i) & 1:
                w |= (1 << m)
        wires.append(w)
    return wires

def _run_network(net, wires):
    """Run network on wire integers. Returns new list of wire ints."""
    x = list(wires)
    for i, j in net:
        a = x[i]
        b = x[j]
        # swap = bits where a=1, b=0
        swap = a & ~b
        x[i] = a ^ swap  # clear the 1s that get swapped out
        x[j] = b | swap  # set the 1s that get swapped in
    return x

def _count_unsorted(net, wires):
    """Count unsorted masks after running net."""
    x = _run_network(net, wires)
    total = 0
    for i in range(N):
        total += bin(x[i]).count('1')
    return TOTAL_MASKS * N - total

def _is_sorted(net, wires):
    """Check if net sorts all masks."""
    return _count_unsorted(net, wires) == 0

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    random.seed(42)
    
    # Precompute wire integers
    wires = _init_wire_ints()
    
    # Start with seed
    best_net = _build_seed()
    best_unsorted = _count_unsorted(best_net, wires)
    best_valid = best_net[:] if best_unsorted == 0 else None
    best_score = len(best_net) if best_valid else 999999
    
    # All possible comparator positions
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Simulated annealing
    net = best_net[:]
    cur_unsorted = best_unsorted
    T = 50.0  # initial temperature
    T_min = 0.1
    cooling = 0.9995
    
    step = 0
    while time.time() < deadline - 2:
        step += 1
        if step % 100 == 0:
            T = max(T_min, T * cooling)
        
        # Choose a random move
        move_type = random.random()
        trial = net[:]
        valid_move = False
        
        if move_type < 0.35 and len(net) > 0:
            # Remove a comparator
            idx = random.randrange(len(net))
            trial.pop(idx)
            valid_move = True
        elif move_type < 0.70 and len(net) < 4096:
            # Add a comparator
            pos = random.choice(all_comps)
            ins_idx = random.randrange(len(net) + 1)
            trial.insert(ins_idx, pos)
            valid_move = True
        elif move_type < 0.85 and len(net) >= 2:
            # Swap two comparators
            i1 = random.randrange(len(net))
            i2 = random.randrange(len(net))
            if i1 != i2:
                trial[i1], trial[i2] = trial[i2], trial[i1]
                valid_move = True
        else:
            # Replace a comparator
            if len(net) > 0:
                idx = random.randrange(len(net))
                trial[idx] = random.choice(all_comps)
                valid_move = True
        
        if not valid_move:
            continue
        
        new_unsorted = _count_unsorted(trial, wires)
        delta = new_unsorted - cur_unsorted
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta <= 0:
            net = trial
            cur_unsorted = new_unsorted
            if cur_unsorted == 0 and len(net) < best_score:
                best_score = len(net)
                best_valid = net[:]
        else:
            # Accept worsening with probability
            if random.random() < (2.718281828 ** (-delta / T)):
                net = trial
                cur_unsorted = new_unsorted
        
        # Periodically try aggressive removal when we find a valid solution
        if cur_unsorted == 0 and step % 50 == 0:
            # Try to remove more comparators greedily
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                trial2 = net[:idx] + net[idx + 1:]
                cnt2 = _count_unsorted(trial2, wires)
                if cnt2 == 0:
                    net = trial2
                    cur_unsorted = 0
                    if len(net) < best_score:
                        best_score = len(net)
                        best_valid = net[:]
                    break  # restart scan
    
    if best_valid is not None:
        return best_valid
    else:
        return _build_seed()

# EVOLVE-BLOCK-END
