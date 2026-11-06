"""Rewritten program: Optimized Local Search for Sorting Networks."""
import time
import numpy as np
import random

N = 17
DEADLINE = 85.0  # Safety margin for 90s limit
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _make_all_inputs():
    """All 2^N boolean inputs as uint8 array shape (2^N, N)."""
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.uint8)

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def simulate_network(comps, initial_data=None):
    """Simulate network on all boolean inputs. Returns final uint8 array (2^N, N)."""
    if initial_data is None:
        data = _get_all_inputs().copy()
    else:
        data = initial_data.copy()
    
    for a, b in comps:
        # Vectorized min/max
        lo = np.minimum(data[:, a], data[:, b])
        hi = np.maximum(data[:, a], data[:, b])
        data[:, a] = lo
        data[:, b] = hi
    return data

def is_sorted_network(comps):
    """Check if network sorts all boolean inputs (zero-one principle)."""
    final = simulate_network(comps)
    # Check if each row is non-decreasing
    # final[:, :-1] <= final[:, 1:]
    return np.all(final[:, :-1] <= final[:, 1:])

def generate_oetn():
    """Generate Odd-Even Transposition Network for N wires."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps

def construct():
    """Construct a sorting network using local search."""
    global START_TIME
    START_TIME = time.time()
    
    # Start with OETN
    comps = generate_oetn()
    
    # Pre-generate all possible pairs for replacement moves
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Phase 1: Greedy Removal
    # We repeatedly try to remove comparators until no more can be removed.
    # We do this in passes.
    removal_passes = 0
    max_removal_passes = 100 # Safety cap, usually converges fast
    
    while _time_left() > 5.0 and removal_passes < max_removal_passes:
        improved = False
        # Iterate through indices in reverse to handle index shifting easily?
        # Or just break and restart if removed.
        # Let's iterate forward and break on success.
        for idx in range(len(comps)):
            if _time_left() < 1.0:
                break
            
            # Try removing comps[idx]
            # Optimization: If comps[idx] is identical to comps[idx-1] or comps[idx+1], 
            # removing it is very likely safe if the network was sorted before.
            # But we must verify.
            
            new_comps = comps[:idx] + comps[idx + 1:]
            
            # Fast check
            if is_sorted_network(new_comps):
                comps = new_comps
                improved = True
                break # Restart loop with new list
        
        if not improved:
            break
        removal_passes += 1
    
    # Phase 2: Local Search with Swaps and Replacements
    # We try to swap adjacent comparators or replace a comparator with a random one.
    # This helps reorder the network to allow further removals or just find a better structure.
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while _time_left() > 2.0 and no_improvement_count < max_no_improvement:
        improved = False
        
        # 1. Try swapping adjacent comparators
        # This is often more effective than random swaps as it preserves local structure.
        if len(comps) > 1:
            # Randomly pick a position to try swapping
            # To be efficient, we try a few random swaps per iteration
            for _ in range(5):
                if _time_left() < 1.0:
                    break
                if len(comps) <= 1:
                    break
                i = random.randint(0, len(comps) - 2)
                # Swap comps[i] and comps[i+1]
                comps[i], comps[i+1] = comps[i+1], comps[i]
                
                if is_sorted_network(comps):
                    improved = True
                    # If improved, we might be able to remove more now?
                    # Let's just continue with this new state.
                    break
                else:
                    # Revert
                    comps[i], comps[i+1] = comps[i+1], comps[i]
            
            if improved:
                no_improvement_count = 0
                # After a swap, it might be possible to remove a comparator again.
                # So we go back to Phase 1 for a bit.
                # Let's do a quick removal pass.
                for idx in range(len(comps)):
                    if _time_left() < 1.0:
                        break
                    new_comps = comps[:idx] + comps[idx + 1:]
                    if is_sorted_network(new_comps):
                        comps = new_comps
                        break
                continue
        
        # 2. Try replacing a random comparator with a random other comparator
        if len(comps) > 0:
            for _ in range(5):
                if _time_left() < 1.0:
                    break
                idx = random.randint(0, len(comps) - 1)
                new_comp = random.choice(all_pairs)
                if comps[idx] == new_comp:
                    continue
                
                old_comp = comps[idx]
                comps[idx] = new_comp
                
                if is_sorted_network(comps):
                    improved = True
                    # Keep replacement
                    # Try to remove again
                    for r_idx in range(len(comps)):
                        if _time_left() < 1.0:
                            break
                        new_comps = comps[:r_idx] + comps[r_idx + 1:]
                        if is_sorted_network(new_comps):
                            comps = new_comps
                            break
                    break
                else:
                    # Revert
                    comps[idx] = old_comp
            
            if improved:
                no_improvement_count = 0
                continue
        
        if not improved:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            
    return comps

if __name__ == "__main__":
    result = construct()
    print(f"Comparators: {len(result)}")
