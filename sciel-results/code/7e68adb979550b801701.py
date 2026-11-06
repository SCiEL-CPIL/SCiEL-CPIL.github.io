"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
SEED_COMPARATORS = 153  # odd-even transposition seed size

# EVOLVE-BLOCK-START
def _apply_network(net, vec):
    """Apply a list of (i,j) comparators to a vector (numpy array)."""
    for i, j in net:
        if vec[i] > vec[j]:
            vec[i], vec[j] = vec[j], vec[i]
    return vec

def _apply_network_batch(net, vecs):
    """Apply network to a 2D array of vectors (shape: [n_vecs, N]). Returns sorted 2D array."""
    result = vecs.copy()
    for i, j in net:
        # For each vector, compare positions i and j
        vals_i = result[:, i]
        vals_j = result[:, j]
        # min goes to i, max goes to j
        result[:, i] = np.minimum(vals_i, vals_j)
        result[:, j] = np.maximum(vals_i, vals_j)
    return result

def _check_network_batch(net, test_vecs_2d):
    """Check if network sorts all vectors in test_vecs_2d (shape [n_vecs, N]).
    Returns True if all sorted."""
    result = _apply_network_batch(net, test_vecs_2d)
    diffs = np.diff(result, axis=1)
    return np.all(diffs >= 0)

def _is_sorted_network(net, n_vecs=None, seed_vecs=None):
    """Check if network sorts all given vectors. Returns True if sorted."""
    if seed_vecs is None:
        # Use a set of random vectors
        rng = np.random.default_rng(42)
        seed_vecs = [rng.integers(0, 1000, size=N) for _ in range(n_vecs or 200)]
    for v in seed_vecs:
        vv = v.copy()
        _apply_network(net, vv)
        if not np.all(np.diff(vv) >= 0):
            return False
    return True

def _depth(net):
    """Compute parallel depth (longest path)."""
    if not net:
        return 0
    depth = np.zeros(N, dtype=int)
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return int(np.max(depth))

def _random_vector_check(net, n_tests=500):
    """Fast random check."""
    rng = np.random.default_rng(12345)
    for _ in range(n_tests):
        v = rng.integers(0, 10000, size=N)
        vv = v.copy()
        _apply_network(net, vv)
        if not np.all(np.diff(vv) >= 0):
            return False, v
    return True, None

def _seed_network():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _build_batcher(n):
    """Build a Batcher odd-even merge sort network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # Recursively sort two halves
    left = _build_batcher(half)
    right = [(i + half, j + half) for i, j in _build_batcher(half)]
    # Odd-even merge
    merge = []
    for phase in range(n):
        parity = phase % 2
        for i in range(parity, n - 1, 2):
            merge.append((i, i + 1))
    return left + right + merge

def _merge_two(nets_a, nets_b, n):
    """Merge two sorting networks for n wires using odd-even merge pattern."""
    # This is a simplified approach; proper merge requires bitonic-like structure
    # For now, just concatenate
    return nets_a + nets_b

def _random_comparator(n, rng):
    i = rng.integers(0, n - 1)
    j = rng.integers(i + 1, n)
    return (int(i), int(j))

def _generate_test_vectors(n_random=400, n_perms=80):
    """Generate a comprehensive set of test vectors as a 2D numpy array."""
    rng = np.random.default_rng(777)
    vecs = []
    # Random vectors
    vecs.append(rng.integers(0, 10000, size=(n_random, N)))
    # Structured vectors
    vecs.append(np.array([np.arange(N, 0, -1)]))  # reverse
    vecs.append(np.array([np.arange(N)]))  # sorted
    vecs.append(np.array([np.array([N-1-i for i in range(N)])]))  # reverse again
    # Random permutations
    perm_vecs = []
    for _ in range(n_perms):
        perm = np.random.default_rng(42 + _).permutation(N)
        perm_vecs.append(perm)
    vecs.append(np.array(perm_vecs))
    # All-ones, all-zeros, two-value patterns
    vecs.append(np.ones((10, N), dtype=int))
    vecs.append(np.zeros((10, N), dtype=int))
    rng2 = np.random.default_rng(999)
    vecs.append(rng2.integers(0, 2, size=(50, N)))  # binary vectors
    return np.vstack(vecs)

def _local_search(net, max_iters=2000, time_limit=40.0, rng_seed=0, test_vecs_2d=None):
    """Hill-climbing: try to remove or swap comparators using batched verification."""
    rng = np.random.default_rng(rng_seed)
    net = list(net)
    best_net = list(net)
    best_len = len(net)
    start_time = time.time()
    
    if test_vecs_2d is None:
        test_vecs_2d = _generate_test_vectors()
    
    def check(net_list):
        return _check_network_batch(net_list, test_vecs_2d)
    
    # First verify current net
    if not check(net):
        return best_net
    
    # Systematic removal sweep: try removing each comparator
    for sweep in range(3):
        if time.time() - start_time > time_limit:
            break
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            if len(net) <= 68:
                break
            new_net = net[:idx] + net[idx+1:]
            if check(new_net):
                net = new_net
                changed = True
                if len(net) < best_len:
                    best_len = len(net)
                    best_net = list(net)
        if not changed:
            break
    
    # Random local search
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if len(net) <= 68:  # lower bound
            break
            
        # Try random modifications
        op = rng.integers(0, 4)
        if op == 0 and len(net) > 2:
            # Remove a random comparator
            idx = rng.integers(0, len(net))
            new_net = net[:idx] + net[idx+1:]
            if check(new_net):
                net = new_net
                if len(net) < best_len:
                    best_len = len(net)
                    best_net = list(net)
        elif op == 1:
            # Swap two adjacent comparators
            if len(net) < 2:
                continue
            idx = rng.integers(0, len(net) - 1)
            new_net = list(net)
            new_net[idx], new_net[idx+1] = new_net[idx+1], new_net[idx]
            if check(new_net):
                net = new_net
        elif op == 2:
            # Replace a random comparator with a different one
            if not net:
                continue
            idx = rng.integers(0, len(net))
            new_i, new_j = _random_comparator(N, rng)
            new_net = list(net)
            new_net[idx] = (new_i, new_j)
            if check(new_net):
                net = new_net
                if len(net) < best_len:
                    best_len = len(net)
                    best_net = list(net)
        elif op == 3 and len(net) > 2:
            # Try moving a comparator to a different position
            idx = rng.integers(0, len(net))
            new_idx = rng.integers(0, len(net))
            if idx == new_idx or idx == new_idx - 1:
                continue
            comp = net[idx]
            new_net = net[:idx] + net[idx+1:]
            # Adjust new_idx if necessary
            if new_idx > idx:
                new_idx -= 1
            new_net = new_net[:new_idx] + [comp] + new_net[new_idx:]
            if check(new_net):
                net = new_net
    
    return best_net

def _greedy_add(net, n_add=10, time_limit=10.0):
    """Try to add useful comparators (not directly helpful for minimization, but for restructuring)."""
    pass  # Not used in minimization

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill
    
    # Start with seed network
    seed_net = _seed_network()
    best_net = list(seed_net)
    best_len = len(seed_net)
    
    # Pre-generate test vectors for batched verification
    test_vecs_2d = _generate_test_vectors(n_random=500, n_perms=100)
    
    # Phase 1: Local search from seed
    t1 = time.time()
    net1 = _local_search(seed_net, max_iters=10000, time_limit=30.0, rng_seed=0, test_vecs_2d=test_vecs_2d)
    if len(net1) < best_len:
        best_net = net1
        best_len = len(net1)
    
    # Phase 2: Try another random restart with perturbation
    if time.time() < deadline - 25.0:
        # Create a perturbed version of seed
        rng = np.random.default_rng(123)
        perturbed = list(seed_net)
        # Remove some random comparators and see if still valid
        for _ in range(30):
            if len(perturbed) <= 80:
                break
            idx = rng.integers(0, len(perturbed))
            trial = perturbed[:idx] + perturbed[idx+1:]
            if _check_network_batch(trial, test_vecs_2d):
                perturbed = trial
        net2 = _local_search(perturbed, max_iters=5000, time_limit=20.0, rng_seed=1, test_vecs_2d=test_vecs_2d)
        if len(net2) < best_len:
            best_net = net2
            best_len = len(net2)
    
    # Phase 3: Final local search from best
    if time.time() < deadline - 8.0:
        remaining_time = max(3.0, deadline - 8.0 - time.time())
        net3 = _local_search(best_net, max_iters=10000, time_limit=remaining_time, rng_seed=99, test_vecs_2d=test_vecs_2d)
        if len(net3) < best_len:
            best_net = net3
            best_len = len(net3)
    
    # Phase 4: Another restart from a different perturbation
    if time.time() < deadline - 5.0:
        rng2 = np.random.default_rng(456)
        perturbed2 = list(best_net)
        # Randomly swap some comparators
        for _ in range(10):
            if len(perturbed2) < 2:
                break
            idx = rng2.integers(0, len(perturbed2) - 1)
            perturbed2[idx], perturbed2[idx+1] = perturbed2[idx+1], perturbed2[idx]
        if _check_network_batch(perturbed2, test_vecs_2d):
            net4 = _local_search(perturbed2, max_iters=3000, time_limit=max(2.0, deadline - 5.0 - time.time()), rng_seed=200, test_vecs_2d=test_vecs_2d)
            if len(net4) < best_len:
                best_net = net4
                best_len = len(net4)
    
    # Verify the best network one more time with more tests
    if time.time() < deadline - 1.0:
        ok, bad_vec = _random_vector_check(best_net, 1000)
        if not ok:
            # Fall back to seed
            best_net = seed_net
    
    return best_net
# EVOLVE-BLOCK-END
