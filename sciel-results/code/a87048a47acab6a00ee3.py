"""Size-efficient sorting networks on 17 wires.
Uses a constructive approach: start from a known-good network (odd-even transposition),
then aggressively prune redundant comparators using the zero-one principle with
vectorized simulation. We also attempt to add a small number of useful comparators
if pruning gets stuck, but primarily focus on aggressive removal.
"""
import time
import numpy as np

N = 17
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _apply_comparators(network, x):
    """Apply a list of comparators to a boolean array x (shape (1, N) or (M, N))."""
    x = x.copy()
    for i, j in network:
        # x[:, i] = min, x[:, j] = max
        # For 0/1: min = x_i AND x_j, max = x_i OR x_j
        # But careful: we need simultaneous update? No, sequential.
        # Actually, compare-exchange: min goes to i, max to j.
        # If x[i] > x[j], swap.
        # In 0/1: if x[i]=1, x[j]=0 -> after: x[i]=0, x[j]=1.
        # min(x_i, x_j) = x_i & x_j
        # max(x_i, x_j) = x_i | x_j
        # Wait, this is only true if we consider the values.
        # Let's just do:
        # new_i = x[i] & x[j]
        # new_j = x[i] | x[j]
        # But this is wrong if we do it in-place without temp.
        # Let's use temp.
        ti = x[:, i].copy()
        tj = x[:, j].copy()
        x[:, i] = np.minimum(ti, tj)
        x[:, j] = np.maximum(ti, tj)
    return x

def _is_sorted(x):
    """Check if boolean array is sorted (non-decreasing). Shape (M, N)."""
    # A sorted 0/1 array has all 0s followed by all 1s.
    # So x[:, k] <= x[:, k+1] for all k.
    return np.all(x[:, :-1] <= x[:, 1:], axis=1)

def _verify(network):
    """Verify network sorts all 2^17 boolean inputs. Returns True if valid."""
    if not network:
        return False
    # Generate all 2^17 boolean inputs
    # 131072 rows, 17 cols
    # Use np.arange and bit manipulation
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for k in range(N):
        all_inputs[:, k] = (np.arange(1 << N) >> k) & 1
    
    result = _apply_comparators(network, all_inputs)
    return bool(np.all(_is_sorted(result)))

def _verify_sample(network, sample_size=10000, seed=42):
    """Quick verification on a random sample. Returns True if sample is sorted."""
    rng = np.random.RandomState(seed)
    samples = rng.randint(0, 2, size=(sample_size, N)).astype(np.uint8)
    result = _apply_comparators(network, samples)
    return bool(np.all(_is_sorted(result)))

def _remove_redundant(network, deadline):
    """Try to remove each comparator if the network remains valid.
    Uses full verification for safety, but with early exit on time."""
    n = len(network)
    # Try removing from the end first? Or random order?
    # Let's try in reverse order, as later comparators are sometimes more redundant.
    # Actually, let's try all positions.
    # To be efficient, we can first do a quick sample check, then full check.
    
    removed = 0
    # We'll iterate through indices, trying to remove each one.
    # If removal is valid, we keep it removed and continue with the shorter list.
    # This is greedy and may miss combinations, but it's fast.
    
    i = 0
    while i < len(network):
        if time.time() > deadline - 5:
            break
        # Try removing network[i]
        candidate = network[:i] + network[i+1:]
        if not candidate:
            i += 1
            continue
        # Quick sample check
        if _verify_sample(candidate, sample_size=5000, seed=i+1):
            # Full verification
            if _verify(candidate):
                network = candidate
                removed += 1
                # Don't increment i, because the list is shorter now
            else:
                i += 1
        else:
            i += 1
    return network, removed

def _add_useful_comparators(network, deadline):
    """Try adding comparators that reduce the number of inversions or improve sorting.
    This is a heuristic: for each pair (i,j), if adding it allows removal of others,
    it might be useful. But that's complex.
    Simpler: just try to see if adding a comparator makes the network valid when
    we remove some others. This is a local search.
    For now, skip this to focus on pruning."""
    return network

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with odd-even transposition network
    # For N wires, odd-even transposition has N phases.
    # Phase p: compare (p%2, p%2+1), (p%2+2, p%2+3), ...
    # Total comparators: N * ceil(N/2)
    # For N=17: 17 * 9 = 153? Wait, the seed says 136.
    # Let's check the seed code:
    # return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    # For N=17:
    # phase 0: i in range(0, 16, 2) -> 0,2,4,6,8,10,12,14 -> 8 comparators
    # phase 1: i in range(1, 16, 2) -> 1,3,5,7,9,11,13,15 -> 8 comparators
    # ...
    # So 17 phases * 8 = 136. Correct.
    
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial network
    if not _verify(network):
        # Should not happen, but just in case
        return network
    
    # Aggressively prune
    network, _ = _remove_redundant(network, deadline)
    
    # Try a second pass of pruning (sometimes removing one enables another)
    if time.time() < deadline - 10:
        network, _ = _remove_redundant(network, deadline)
    
    # If we have time, try a third pass
    if time.time() < deadline - 5:
        network, _ = _remove_redundant(network, deadline)
    
    # Final verification
    if not _verify(network):
        # This should not happen if pruning is correct, but just in case
        # Return the original network
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    return network
