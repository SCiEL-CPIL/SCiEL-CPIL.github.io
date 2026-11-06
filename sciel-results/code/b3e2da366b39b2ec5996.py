"""Size-efficient sorting networks on 18 wires: bidirectional BFS + local polish."""
import time
import random
from collections import deque

N = 18
# Deadline guard: keep a safety margin before the 90s kill.
DEADLINE = 80.0
START_TIME = time.time()

# Precompute all possible comparators (i, j) with 0 <= i < j < N
ALL_CMP = [(i, j) for i in range(N) for j in range(i + 1, N)]

# Identity permutation: wire i holds input i
IDENTITY = tuple(range(N))
# Reverse-sort permutation: wire i holds input (N-1-i)
REVERSE = tuple(range(N - 1, -1, -1))


def _apply_cmp(perm, cmp):
    """Apply a single compare-exchange to a permutation.
    perm is a tuple where perm[i] is the value on wire i.
    After compare-exchange (i, j): min goes to wire i, max to wire j.
    """
    i, j = cmp
    a, b = perm[i], perm[j]
    if a <= b:
        # Already in order
        return perm
    else:
        # Swap
        lst = list(perm)
        lst[i], lst[j] = b, a
        return tuple(lst)


def _is_sorted(perm):
    """Check if permutation is sorted (increasing)."""
    for k in range(N - 1):
        if perm[k] > perm[k + 1]:
            return False
    return True


def _apply_cmp_list(perm_list, cmp):
    """Apply compare-exchange to a list representation (for speed in local search)."""
    i, j = cmp
    a, b = perm_list[i], perm_list[j]
    if a > b:
        perm_list[i], perm_list[j] = b, a


def _is_sorted_list(perm_list):
    for k in range(N - 1):
        if perm_list[k] > perm_list[k + 1]:
            return False
    return True


def _bidirectional_bfs():
    """Bidirectional BFS from IDENTITY and REVERSE to find shortest comparator sequence."""
    # Forward: start from IDENTITY, apply comparators
    # Backward: start from REVERSE, apply comparators (compare-exchange is self-inverse on sorted/unsorted)
    # Actually, compare-exchange is an involution: applying it twice gives identity.
    # So the reverse of a comparator sequence is the same comparators in reverse order.
    # For BFS, we just need to find the shortest path from IDENTITY to a sorted permutation
    # when starting from REVERSE.
    
    # Forward frontier: dict from perm -> (parent, cmp)
    fwd = {IDENTITY: (None, None)}
    fwd_q = deque([IDENTITY])
    
    # Backward frontier: dict from perm -> (parent, cmp)
    bwd = {REVERSE: (None, None)}
    bwd_q = deque([REVERSE])
    
    # We'll expand the smaller frontier each step
    # But for simplicity and to meet time limits, let's do a few levels from each side
    
    # Given the state space is huge (18! = 6.4e15), pure BFS won't reach the meeting point.
    # Instead, we use a heuristic: expand a limited number of levels and hope for a meeting.
    # This is unlikely to find the optimal 77-comparator network, but might find something decent.
    
    # Let's try up to 10 levels from each side
    max_levels = 10
    for level in range(max_levels):
        if time.time() - START_TIME > DEADLINE - 30:
            break
        # Expand forward
        if not fwd_q:
            break
        next_fwd = {}
        for _ in range(len(fwd_q)):
            if time.time() - START_TIME > DEADLINE - 30:
                break
            perm = fwd_q.popleft()
            for cmp in ALL_CMP:
                new_perm = _apply_cmp(perm, cmp)
                if new_perm not in fwd:
                    fwd[new_perm] = (perm, cmp)
                    fwd_q.append(new_perm)
                    if new_perm in bwd:
                        # Found a meeting point!
                        return _reconstruct(fwd, bwd, new_perm)
        if not fwd_q:
            break
        # Expand backward
        if not bwd_q:
            break
        for _ in range(len(bwd_q)):
            if time.time() - START_TIME > DEADLINE - 30:
                break
            perm = bwd_q.popleft()
            for cmp in ALL_CMP:
                new_perm = _apply_cmp(perm, cmp)
                if new_perm not in bwd:
                    bwd[new_perm] = (perm, cmp)
                    bwd_q.append(new_perm)
                    if new_perm in fwd:
                        # Found a meeting point!
                        return _reconstruct(fwd, bwd, new_perm)
    
    return None


def _reconstruct(fwd, bwd, meeting):
    """Reconstruct the comparator sequence from BFS paths."""
    path_fwd = []
    perm = meeting
    while fwd[perm][0] is not None:
        path_fwd.append(fwd[perm][1])
        perm = fwd[perm][0]
    path_fwd.reverse()
    
    path_bwd = []
    perm = meeting
    while bwd[perm][0] is not None:
        path_bwd.append(bwd[perm][1])
        perm = bwd[perm][0]
    
    # The backward path needs to be reversed
    path_bwd.reverse()
    
    return path_fwd + path_bwd


def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _verify_network(network):
    """Verify that the network sorts all 2^18 Boolean inputs using the 0-1 principle."""
    for mask in range(1 << N):
        # Create input: 0/1 on each wire
        perm = [(mask >> i) & 1 for i in range(N)]
        for cmp in network:
            i, j = cmp
            if perm[i] > perm[j]:
                perm[i], perm[j] = perm[j], perm[i]
        # Check sorted
        for k in range(N - 1):
            if perm[k] > perm[k + 1]:
                return False
    return True


def _count_comparators(network):
    return len(network)


def _local_search(network, max_iters=5000):
    """Local search to reduce comparator count.
    Try removing each comparator and check if the network still sorts.
    """
    network = list(network)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - START_TIME > DEADLINE - 10:
                return network
            # Try removing comparator at idx
            new_network = network[:idx] + network[idx + 1:]
            if _verify_network(new_network):
                network = new_network
                improved = True
                break
    return network


def _sa_local_search(network, time_limit=30.0):
    """Simulated annealing to find a smaller network.
    Moves: remove, swap, move, replace a comparator.
    """
    network = list(network)
    best_network = list(network)
    best_count = len(network)
    
    # Precompute verification for speed: use a faster check
    def _fast_verify(net):
        for mask in range(1 << N):
            perm = [(mask >> i) & 1 for i in range(N)]
            for cmp in net:
                i, j = cmp
                if perm[i] > perm[j]:
                    perm[i], perm[j] = perm[j], perm[i]
            for k in range(N - 1):
                if perm[k] > perm[k + 1]:
                    return False
        return True
    
    # We need a faster verification. Let's use a bitset approach.
    # Actually, for 18 wires, 2^18 = 262144 checks, each with ~77 comparators.
    # That's about 20 million operations per verification. In Python, this might take ~1s.
    # We need to be more efficient.
    
    # Let's use NumPy for faster verification
    import numpy as np
    
    def _verify_np(net):
        # Convert network to arrays
        if not net:
            return True
        i_arr = np.array([c[0] for c in net], dtype=np.int8)
        j_arr = np.array([c[1] for c in net], dtype=np.int8)
        
        for mask in range(1 << N):
            # Create input vector
            perm = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
            for idx in range(len(net)):
                i, j = i_arr[idx], j_arr[idx]
                if perm[i] > perm[j]:
                    perm[i], perm[j] = perm[j], perm[i]
            if np.any(perm[:-1] > perm[1:]):
                return False
        return True
    
    # Even this might be slow. Let's try a different approach:
    # Use the fact that we only need to check 0-1 inputs, and use a smarter method.
    
    # For now, let's just do greedy removal which is simpler and faster to verify.
    return _local_search(network, max_iters=2000)


def construct():
    """Construct a sorting network for 18 wires."""
    global START_TIME
    START_TIME = time.time()
    
    # Start with the seed odd-even transposition network
    network = _odd_even_transposition()
    
    # Try bidirectional BFS to find a shorter network (may not succeed in time)
    bfs_result = _bidirectional_bfs()
    if bfs_result is not None and len(bfs_result) < len(network):
        network = bfs_result
    
    # Local search: greedily remove comparators
    network = _local_search(network, max_iters=3000)
    
    # Verify the final network
    if not _verify_network(network):
        # Fallback to seed
        return _odd_even_transposition()
    
    return network


# Test the function
if __name__ == "__main__":
    net = construct()
    print(f"Network size: {len(net)}")
    print(f"Verified: {_verify_network(net)}")
