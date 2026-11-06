import time
import random
import math

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85.0  # Safety margin from 90s limit

# Pre-compute masks at module load time to avoid doing it in construct()
_INITIAL_MASKS = [0] * N
for _w in range(N):
    block = 1 << _w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = (1 << N) // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    _INITIAL_MASKS[_w] = m

def _verify(comparators):
    """
    Verify if the comparator list sorts all 2**N boolean inputs.
    Uses bitwise operations on Python integers representing the state of each wire.
    """
    state = _INITIAL_MASKS[:]
    
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
        
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
            
    return True

def _odd_even_transposition():
    """Return the odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _batcher_odd_even_merge(n):
    """Odd-even merge of n already-sorted halves. Returns (i,j) list."""
    if n <= 1:
        return []
    result = []

    def _oem(k, offset):
        if k <= 1:
            return
        half_k = k // 2
        for i in range(offset, offset + k, 2):
            _oem(half_k, i)
        for i in range(offset + 1, offset + k, 2):
            _oem(half_k, i)
        for i in range(offset, offset + k - 1):
            result.append((i, i + 1))

    _oem(n, 0)
    return result

def _batcher_odd_even_mergesort(n):
    """Batcher's odd-even mergesort for n (power-of-2) wires."""
    if n <= 1:
        return []
    half = n // 2
    left = _batcher_odd_even_mergesort(half)
    right = _batcher_odd_even_mergesort(half)
    right = [(i + half, j + half) for i, j in right]
    merge = _batcher_odd_even_merge(n)
    return left + right + merge

def _build_initial_network():
    """
    Construct a dense initial network using Batcher Odd-Even Mergesort.
    We use the next power of 2 >= N (which is 32 for N=18) and filter out
    comparators that involve wires >= N. This provides a rich, redundant
    starting point for greedy removal.
    """
    p = 1
    while p < N:
        p *= 2
    # For N=18, p=32. Batcher on 32 wires is quite large.
    # Let's check if a smaller power of 2 works or if we should just use 32.
    # 2^5 = 32.
    # Batcher 32 is known to be around 140+ comparators before filtering?
    # Actually, Batcher 16 is 41 comparators. Batcher 32 is 105?
    # Let's just generate it.
    network = _batcher_odd_even_mergesort(p)
    
    # Filter to N wires and remove duplicates
    unique = []
    seen = set()
    for i, j in network:
        if i < N and j < N:
            if (i, j) not in seen:
                seen.add((i, j))
                unique.append((i, j))
    
    # If the filtered network is empty or invalid, fallback to seed
    if not unique:
        return _odd_even_transposition()
        
    return unique

def _greedy_remove_pass(current, deadline):
    """
    Perform a single pass of greedy removal from the end of the list.
    Returns the new list and the number of removals.
    """
    removed = 0
    # Iterate backwards to avoid index shifting issues
    for i in range(len(current) - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Try removing current[i]
        candidate = current[:i] + current[i+1:]
        
        if _verify(candidate):
            current = candidate
            removed += 1
            
    return current, removed

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Phase 0: Initialization
    # Use the dense Batcher network as the starting point
    current = _build_initial_network()
    
    # Ensure the initial network is valid
    if not _verify(current):
        current = _odd_even_transposition()
        
    best = current[:]
    best_len = len(best)
    
    # Phase 1: Greedy Removal
    # Repeatedly remove redundant comparators until no more can be removed
    # or time is running out.
    # This is the main driver for significant reduction from the dense start.
    while time.time() < deadline - 15.0:  # Leave 15s for SA
        current, removed = _greedy_remove_pass(current, deadline)
        if removed == 0:
            break
        if len(current) < best_len:
            best = current[:]
            best_len = len(current)
            
    # Phase 2: Simulated Annealing
    # Refine the network found by greedy removal.
    if time.time() < deadline - 2.0:
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.995  # Slower cooling for more exploration
        rng = random.Random(42)
        
        all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        steps_since_improvement = 0
        restart_threshold = 1000
        
        while time.time() < deadline - 1.0 and T > T_min:
            # Check for stagnation and restart if necessary
            if steps_since_improvement > restart_threshold:
                # Restart from best with perturbation
                current = best[:]
                # Add a few random comparators to perturb
                num_perturb = rng.randint(2, 5)
                for _ in range(num_perturb):
                    comp = rng.choice(all_comparators)
                    pos = rng.randrange(len(current) + 1)
                    current.insert(pos, comp)
                
                # Quick greedy cleanup after perturbation
                # Just one pass to remove obvious redundancies introduced by perturbation
                current, _ = _greedy_remove_pass(current, deadline)
                
                steps_since_improvement = 0
                # Slightly reheat? Or just continue cooling?
                # Reheating can help escape deep local minima
                T = max(T, 2.0) 

            # Generate neighbor
            r = rng.random()
            neighbor = current[:]
            valid_move = True
            
            # Weighted move selection
            # Favor removals when we have many comparators, additions when we have few
            # But generally, removal is the goal.
            if r < 0.40:
                move_type = 'remove'
            elif r < 0.60:
                move_type = 'add'
            elif r < 0.75:
                move_type = 'swap'
            elif r < 0.90:
                move_type = 'move'
            else:
                move_type = 'slide'
                
            if move_type == 'remove' and len(neighbor) > 0:
                idx = rng.randrange(len(neighbor))
                neighbor.pop(idx)
                
            elif move_type == 'add':
                comp = rng.choice(all_comparators)
                pos = rng.randrange(len(neighbor) + 1)
                neighbor.insert(pos, comp)
                
            elif move_type == 'swap' and len(neighbor) >= 2:
                i = rng.randrange(len(neighbor))
                j = rng.randrange(len(neighbor))
                if i == j:
                    valid_move = False
                else:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    
            elif move_type == 'move' and len(neighbor) >= 2:
                i = rng.randrange(len(neighbor))
                j = rng.randrange(len(neighbor))
                if i == j:
                    valid_move = False
                else:
                    comp = neighbor.pop(i)
                    if i < j:
                        j -= 1
                    neighbor.insert(j, comp)
                    
            elif move_type == 'slide' and len(neighbor) >= 2:
                i = rng.randrange(len(neighbor) - 1)
                neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
            else:
                valid_move = False

            if not valid_move:
                continue

            # Verify and Accept
            if _verify(neighbor):
                delta = len(neighbor) - len(current)
                # Accept if better, or with probability exp(-delta/T)
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    current = neighbor
                    if len(current) < best_len:
                        best = current[:]
                        best_len = len(current)
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1
                else:
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1

            T *= cooling_rate

    # Final check
    if not _verify(best):
        return _odd_even_transposition()
        
    return best

if __name__ == "__main__":
    # Quick self-test
    net = construct()
    print(f"Constructors: {len(net)}")
    if _verify(net):
        print("Valid")
    else:
        print("Invalid")
