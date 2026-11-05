<<<<<<< SEARCH
"""Shortest superpermutation construction with greedy block reordering."""
import sys
import time
from itertools import permutations

# Deadline guard
_DEADLINE = 115.0  # seconds, safety margin before 120s limit
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def construct(n):
    """Construct a superpermutation on n symbols.
    
    Uses the classic recursive construction with a greedy Eulerian trail
    heuristic on the directed graph of permutation blocks, where edge
    selection prioritizes moves that keep the remaining unvisited edges
    in a single connected component (minimizing disconnected components).
    """
    global _start_time
    _start_time = time.time()
    
    if n <= 1:
        return "1"
    
    # Classic recursive construction
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations from the current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Reorder perms using greedy Eulerian trail heuristic
        if len(perms) <= 500:
            perms = _greedy_eulerian_reorder(perms, m)
        
        # Build new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with end of new
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        # Check deadline
        if _elapsed() > _DEADLINE - 5:
            break
    
    return "".join(map(str, word))


def _greedy_eulerian_reorder(perms, m):
    """Reorder permutation blocks using a greedy Eulerian trail heuristic.
    
    Models the problem as finding a trail in a directed graph where:
    - Nodes are permutation blocks
    - Edges represent valid transitions with overlap costs
    
    The heuristic selects the next edge that:
    1. Maximizes overlap (minimizes added length)
    2. Among ties, minimizes the number of disconnected components in the
       subgraph of unvisited edges (keeping the graph connected)
    
    This is analogous to Hierholzer's algorithm but with a connectivity-aware
    greedy choice that avoids prematurely isolating unvisited edges.
    """
    if len(perms) <= 1:
        return perms
    
    n_perms = len(perms)
    
    # Precompute the full block for each perm
    blocks = [list(p) + [m] + list(p) for p in perms]
    
    # Compute overlap matrix: overlap[i][j] = max overlap of end of blocks[i] with start of blocks[j]
    overlap = [[0] * n_perms for _ in range(n_perms)]
    
    for i in range(n_perms):
        for j in range(n_perms):
            if i == j:
                continue
            bi = blocks[i]
            bj = blocks[j]
            max_ov = min(len(bi), len(bj))
            for o in range(max_ov, -1, -1):
                if o == 0 or bi[len(bi) - o:] == bj[:o]:
                    overlap[i][j] = o
                    break
    
    # Build adjacency structure: for each node, list of (neighbor, overlap_value)
    adj = [[] for _ in range(n_perms)]
    for i in range(n_perms):
        for j in range(n_perms):
            if i != j:
                adj[i].append((j, overlap[i][j]))
    
    # Greedy Eulerian trail with connectivity awareness
    best_order = None
    best_length = float('inf')
    
    # Try multiple starting points
    max_starts = n_perms if n_perms <= 30 else min(15, n_perms)
    
    for start in range(max_starts):
        if _elapsed() > _DEADLINE - 3:
            break
        
        order = [start]
        visited_edges = set()
        remaining = set(range(n_perms)) - {start}
        current_length = len(blocks[start])
        
        while remaining:
            last = order[-1]
            
            # Score each candidate next node
            best_next = None
            best_score = None
            
            for j in remaining:
                ov = overlap[last][j]
                
                # Compute connectivity score: how many disconnected components
                # would result in the subgraph of unvisited edges from j's perspective
                # Simplified: count how many remaining nodes are reachable from j
                # via unvisited edges (excluding the edge last->j we're about to use)
                
                # Quick connectivity check: count reachable nodes from j in remaining
                # using BFS/DFS on unvisited edges
                reachable = _count_reachable(j, remaining, adj, visited_edges, last, n_perms)
                
                # Score: primarily maximize overlap, secondarily maximize connectivity
                # We want high overlap AND high connectivity (fewer components)
                score = (ov, reachable)
                
                if best_score is None or score > best_score:
                    best_score = score
                    best_next = j
            
            if best_next is None:
                break
            
            order.append(best_next)
            remaining.remove(best_next)
            visited_edges.add((last, best_next))
            current_length += len(blocks[best_next]) - overlap[last][best_next]
        
        if len(order) == n_perms and current_length < best_length:
            best_length = current_length
            best_order = order
    
    if best_order is None:
        # Fallback: original order
        return perms
    
    # Apply 2-opt local search to improve the order
    if n_perms <= 100 and _elapsed() < _DEADLINE - 5:
        best_order, best_length = _two_opt_improve(best_order, overlap, blocks, n_perms)
    
    return [perms[i] for i in best_order]


def _count_reachable(start, remaining, adj, visited_edges, from_node, n_perms):
    """Count how many nodes in 'remaining' are reachable from 'start' via unvisited edges.
    
    This is a simplified connectivity metric that helps the greedy heuristic
    avoid choices that would fragment the remaining edge subgraph.
    """
    if not remaining:
        return 0
    
    # BFS from start, only traversing unvisited edges
    visited = set()
    queue = [start]
    visited.add(start)
    
    while queue:
        node = queue.pop(0)
        for neighbor, _ in adj[node]:
            if neighbor in remaining and neighbor not in visited:
                # Check if edge (node, neighbor) is unvisited
                if (node, neighbor) not in visited_edges:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    return len(visited & remaining)


def _two_opt_improve(order, overlap, blocks, n_perms):
    """Improve a permutation order using 2-opt local search.
    
    Tries reversing segments of the order to find better overlaps.
    Returns the improved order and its length.
    """
    current_order = list(order)
    current_length = _order_length(current_order, overlap, blocks)
    
    improved = True
    max_iterations = 50 if n_perms <= 50 else 20
    
    while improved and _elapsed() < _DEADLINE - 3:
        improved = False
        for _ in range(max_iterations):
            if _elapsed() > _DEADLINE - 3:
                break
            
            best_improvement = 0
            best_i, best_j = -1, -1
            
            # Try all 2-opt moves: reverse segment [i, j]
            for i in range(n_perms - 1):
                for j in range(i + 1, n_perms):
                    # Calculate improvement from reversing segment [i, j]
                    # Before: ... a[i-1] -> a[i] ... a[j] -> a[j+1] ...
                    # After:  ... a[i-1] -> a[j] ... a[i] -> a[j+1] ...
                    
                    old_len = 0
                    new_len = 0
                    
                    # Edge entering the segment
                    if i > 0:
                        old_len += len(blocks[current_order[i]]) - overlap[current_order[i-1]][current_order[i]]
                        new_len += len(blocks[current_order[j]]) - overlap[current_order[i-1]][current_order[j]]
                    else:
                        old_len += len(blocks[current_order[i]])
                        new_len += len(blocks[current_order[j]])
                    
                    # Edge leaving the segment
                    if j < n_perms - 1:
                        old_len += len(blocks[current_order[j+1]]) - overlap[current_order[j]][current_order[j+1]]
                        new_len += len(blocks[current_order[i]]) - overlap[current_order[i]][current_order[j+1]]
                    
                    # Internal edges (reversed)
                    for k in range(i + 1, j):
                        old_len += len(blocks[current_order[k]]) - overlap[current_order[k-1]][current_order[k]]
                        new_len += len(blocks[current_order[k]]) - overlap[current_order[k+1]][current_order[k]]
                    
                    improvement = old_len - new_len
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_i, best_j = i, j
            
            if best_improvement > 0:
                # Apply the best 2-opt move
                current_order[best_i:best_j+1] = reversed(current_order[best_i:best_j+1])
                current_length -= best_improvement
                improved = True
    
    return current_order, current_length


def _order_length(order, overlap, blocks):
    """Calculate the total length of concatenating blocks in the given order."""
    total = len(blocks[order[0]])
    for i in range(1, len(order)):
        total += len(blocks[order[i]]) - overlap[order[i-1]][order[i]]
    return total
=======
"""Shortest superpermutation construction with VNS block reordering."""
import sys
import time
import random
from itertools import permutations

# Deadline guard
_DEADLINE = 140.0  # seconds, safety margin before 150s limit
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def construct(n):
    """Construct a superpermutation on n symbols.
    
    Uses the classic recursive construction with Variable Neighborhood Search
    (VNS) on the ordering of (n-1)-permutation blocks at each recursion level.
    VNS uses Or-opt and 2-opt moves to find better block orderings that
    maximize suffix-prefix overlaps between adjacent blocks.
    """
    global _start_time
    _start_time = time.time()
    random.seed(42)
    
    if n <= 1:
        return "1"
    
    # Classic recursive construction
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations from the current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Reorder perms using VNS
        if len(perms) > 1:
            perms = _vns_reorder(perms, m)
        
        # Build new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with end of new
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        # Check deadline
        if _elapsed() > _DEADLINE - 3:
            break
    
    return "".join(map(str, word))


def _compute_overlap_matrix(blocks):
    """Compute overlap matrix efficiently using suffix-prefix matching."""
    n = len(blocks)
    overlap = [[0] * n for _ in range(n)]
    
    for i in range(n):
        bi = blocks[i]
        bi_str = ''.join(map(str, bi))
        for j in range(n):
            if i == j:
                continue
            bj = blocks[j]
            bj_str = ''.join(map(str, bj))
            max_ov = min(len(bi_str), len(bj_str))
            for o in range(max_ov, -1, -1):
                if o == 0 or bi_str[-o:] == bj_str[:o]:
                    overlap[i][j] = o
                    break
    
    return overlap


def _order_cost(order, overlap, block_lens):
    """Calculate total length of concatenating blocks in given order."""
    total = block_lens[order[0]]
    for i in range(1, len(order)):
        total += block_lens[order[i]] - overlap[order[i-1]][order[i]]
    return total


def _vns_reorder(perms, m):
    """Reorder permutation blocks using Variable Neighborhood Search.
    
    Neighborhoods:
    1. Or-opt: move a segment of length 1, 2, or 3 to a different position
    2. 2-opt: reverse a segment
    3. Random restart with perturbation
    """
    n_perms = len(perms)
    
    if n_perms <= 1:
        return perms
    
    # Precompute blocks and overlap matrix
    blocks = [list(p) + [m] + list(p) for p in perms]
    block_lens = [len(b) for b in blocks]
    overlap = _compute_overlap_matrix(blocks)
    
    # Initial solution: greedy nearest neighbor
    best_order = _greedy_nn_order(overlap, block_lens, n_perms)
    best_cost = _order_cost(best_order, overlap, block_lens)
    
    # VNS with multiple restarts
    max_restarts = 5 if n_perms <= 50 else 3
    time_budget = _DEADLINE - _elapsed() - 3
    
    if time_budget <= 0:
        return [perms[i] for i in best_order]
    
    per_restart_budget = time_budget / max_restarts
    
    for restart in range(max_restarts):
        if _elapsed() > _DEADLINE - 3:
            break
        
        # For first restart, use greedy; subsequent restarts use perturbation
        if restart == 0:
            current_order = list(best_order)
        else:
            current_order = _perturb(best_order, n_perms, restart)
        
        current_cost = _order_cost(current_order, overlap, block_lens)
        
        # VNS: cycle through neighborhoods
        no_improvement = 0
        max_no_improve = 10
        
        while no_improvement < max_no_improve and _elapsed() < _DEADLINE - 3:
            improved = False
            
            # Neighborhood 1: Or-opt (move segments of length 1-3)
            for seg_len in [1, 2, 3]:
                if _elapsed() > _DEADLINE - 3:
                    break
                for i in range(n_perms - seg_len + 1):
                    if _elapsed() > _DEADLINE - 3:
                        break
                    for j in range(i + seg_len, n_perms + 1):
                        if j == i + seg_len:
                            continue  # no move
                        # Compute cost delta
                        delta = _oropt_delta(current_order, overlap, block_lens, i, j, seg_len)
                        if delta < 0:
                            # Apply move
                            segment = current_order[i:i+seg_len]
                            current_order = current_order[:i] + current_order[i+seg_len:j] + segment + current_order[j:]
                            current_cost += delta
                            improved = True
                            no_improvement = 0
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Neighborhood 2: 2-opt (reverse segments)
            for i in range(n_perms - 1):
                if _elapsed() > _DEADLINE - 3:
                    break
                for j in range(i + 1, n_perms):
                    delta = _twoopt_delta(current_order, overlap, block_lens, i, j)
                    if delta < 0:
                        # Apply reverse
                        current_order[i:j+1] = reversed(current_order[i:j+1])
                        current_cost += delta
                        improved = True
                        no_improvement = 0
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # No improvement in this neighborhood cycle
            no_improvement += 1
            
            # Update best
            if current_cost < best_cost:
                best_cost = current_cost
                best_order = list(current_order)
            
            # Perturb to escape local optimum
            if no_improvement >= 5:
                current_order = _perturb(best_order, n_perms, restart + no_improvement)
                current_cost = _order_cost(current_order, overlap, block_lens)
                no_improvement = 0
    
    return [perms[i] for i in best_order]


def _greedy_nn_order(overlap, block_lens, n_perms):
    """Greedy nearest neighbor: always pick the next block with max overlap."""
    order = []
    remaining = set(range(n_perms))
    
    # Start with a random block
    start = random.choice(list(remaining))
    order.append(start)
    remaining.remove(start)
    
    while remaining:
        last = order[-1]
        best_next = None
        best_ov = -1
        for j in remaining:
            ov = overlap[last][j]
            if ov > best_ov:
                best_ov = ov
                best_next = j
        order.append(best_next)
        remaining.remove(best_next)
    
    return order


def _perturb(order, n_perms, seed_offset):
    """Perturb an order by applying random swaps and segment reversals."""
    rng = random.Random(seed_offset * 1000 + 7)
    result = list(order)
    num_perturbs = min(5, n_perms // 4)
    for _ in range(num_perturbs):
        choice = rng.random()
        if choice < 0.5:
            # Swap two random positions
            i, j = rng.sample(range(n_perms), 2)
            result[i], result[j] = result[j], result[i]
        else:
            # Reverse a random segment
            i, j = sorted(rng.sample(range(n_perms), 2))
            result[i:j+1] = reversed(result[i:j+1])
    return result


def _oropt_delta(order, overlap, block_lens, i, j, seg_len):
    """Compute cost delta for Or-opt: move segment [i, i+seg_len) to position j."""
    n = len(order)
    if j >= i + seg_len and j <= n:
        # Remove segment from [i, i+seg_len) and insert at position j
        # Old connections affected:
        # - (order[i-1], order[i]) if i > 0
        # - (order[i+seg_len-1], order[i+seg_len]) if i+seg_len < n
        # - (order[j-1], order[j]) if j > 0 and j < n (but j is insertion point)
        
        # Let's compute old and new contributions
        old_cost = 0
        new_cost = 0
        
        # Edge before the segment (if exists)
        if i > 0:
            old_cost += block_lens[order[i]] - overlap[order[i-1]][order[i]]
        
        # Edge after the segment (if exists)
        if i + seg_len < n:
            old_cost += block_lens[order[i+seg_len]] - overlap[order[i+seg_len-1]][order[i+seg_len]]
        
        # Edge at insertion point (if exists)
        if j > 0 and j < n:
            old_cost += block_lens[order[j]] - overlap[order[j-1]][order[j]]
        
        # New connections:
        # - (order[i-1], order[i+seg_len]) if i > 0 and i+seg_len < n
        # - (order[i-1], order[i]) if i > 0 (segment moved away, so i-1 now connects to what was at i+seg_len)
        # Actually, let me think more carefully...
        
        # After removing segment [i, i+seg_len):
        # The element at i becomes what was at i+seg_len (if exists)
        # The element at i+seg_len-1 is removed
        # Then we insert the segment at position j
        
        # This is complex; let me just compute full costs for small segments
        # For efficiency, only compute delta when it's likely beneficial
        
        # Simplified: compute the change in boundary edges
        # Old boundaries: 
        #   left: (i-1 -> i) if i>0
        #   right: (i+seg-1 -> i+seg) if i+seg<n
        #   at j: (j-1 -> j) if 0<j<n
        # New boundaries:
        #   left: (i-1 -> i+seg) if i>0 and i+seg<n
        #   or (i-1 -> first_of_segment) if i>0 and i+seg==n (segment goes to end... no)
        #   Actually after removal, position i has what was at i+seg
        #   After insertion at j, the segment occupies [j, j+seg)
        
        # This is getting complex. Let me use a simpler approach:
        # Just compute the total cost before and after for the affected region
        
        # For now, return 0 (no improvement) to avoid bugs
        pass
    
    return 0  # Conservative: skip Or-opt for now, rely on 2-opt


def _twoopt_delta(order, overlap, block_lens, i, j):
    """Compute cost delta for 2-opt: reverse segment [i, j]."""
    n = len(order)
    if i >= j:
        return 0
    
    # Old edges affected:
    # - (i-1, i) if i > 0
    # - (j, j+1) if j < n-1
    # - Internal edges (k, k+1) for k in [i, j-1]
    
    # New edges:
    # - (i-1, j) if i > 0
    # - (i, j+1) if j < n-1
    # - Internal edges reversed: (k+1, k) for k in [i, j-1]
    
    delta = 0
    
    # Left boundary
    if i > 0:
        delta += (block_lens[order[j]] - overlap[order[i-1]][order[j]])
        delta -= (block_lens[order[i]] - overlap[order[i-1]][order[i]])
    
    # Right boundary
    if j < n - 1:
        delta += (block_lens[order[j+1]] - overlap[order[i]][order[j+1]])
        delta -= (block_lens[order[j+1]] - overlap[order[j]][order[j+1]])
    
    # Internal edges
    for k in range(i, j):
        delta += (block_lens[order[k+1]] - overlap[order[k+1]][order[k]])
        delta -= (block_lens[order[k+1]] - overlap[order[k]][order[k+1]])
    
    return delta
>>>>>>> REPLACE

Now let me also fix the Or-opt delta computation to actually work, since the current stub just returns 0:

