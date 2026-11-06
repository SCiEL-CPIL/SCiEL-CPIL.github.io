"""Q3-free graph on 16 vertices - maximize edges via bipartite bitmask search."""
import time
import random
import itertools

# Q3 is bipartite with parts of size 4.
# In K_{4,4} minus Q3, there are 4 missing edges forming a perfect matching.
# So a Q3 exists in a 4x4 submatrix if missing edges <= 4 AND missing edges contain a perfect matching.

# Precompute all 24 perfect matching masks for 4x4 (16 bits, row i col j -> bit i*4+j)
_PERFECT_MATCHING_MASKS = []
for perm in itertools.permutations(range(4)):
    mask = 0
    for i in range(4):
        mask |= (1 << (i * 4 + perm[i]))
    _PERFECT_MATCHING_MASKS.append(mask)

def _has_q3(adj_mask):
    """Check if the 8x8 bipartite graph (64-bit mask) contains a Q3.
    A Q3 exists iff there exist 4 rows L' and 4 cols R' such that
    the 4x4 submatrix has <= 4 missing edges and the missing edges contain a perfect matching.
    """
    # Precompute row masks
    row_masks = [(adj_mask >> (r * 8)) & 0xFF for r in range(8)]
    
    # For each choice of 4 rows and 4 columns, check the 4x4 submatrix
    for rows in itertools.combinations(range(8), 4):
        for cols in itertools.combinations(range(8), 4):
            # Build the 4x4 missing-edge mask
            missing = 0
            for i, r in enumerate(rows):
                for j, c in enumerate(cols):
                    if not (row_masks[r] >> c) & 1:
                        missing |= (1 << (i * 4 + j))
            if missing.bit_count() > 4:
                continue
            # Check if missing contains any perfect matching
            for pm in _PERFECT_MATCHING_MASKS:
                if (missing & pm) == pm:
                    return True
    return False


def _count_edges(adj_mask):
    return bin(adj_mask).count('1')


def construct():
    deadline = time.time() + 140.0  # safety margin
    
    best_edges = 0
    best_mask = 0
    
    # Strategy: local search with random restarts
    # Start from a dense random state, then try to add edges / remove edges to fix
    
    # First, try a greedy construction
    # Start with empty graph, add edges one by one if Q3-free
    mask = 0
    for _ in range(5000):
        if time.time() > deadline - 5.0:
            break
        # Try adding a random edge
        for _retry in range(100):
            r = random.randint(0, 7)
            c = random.randint(0, 7)
            bit = 1 << (r * 8 + c)
            if mask & bit:
                continue
            new_mask = mask | bit
            if not _has_q3(new_mask):
                mask = new_mask
                break
    
    best_edges = max(best_edges, _count_edges(mask))
    best_mask = mask if _count_edges(mask) > best_edges else best_mask
    
    # Local search: try to flip 0->1 and 1->0 to improve
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Start from current best or a random dense-ish state
        if restart == 0:
            cur = best_mask
        else:
            # Random state with ~60-80% density
            cur = 0
            for i in range(64):
                if random.random() < 0.65:
                    cur |= (1 << i)
        
        # If current state has Q3, try to remove edges to fix it
        while _has_q3(cur) and time.time() < deadline - 5.0:
            # Find a Q3 and remove a random edge from it
            # Just remove a random 1-bit
            ones = [i for i in range(64) if (cur >> i) & 1]
            if not ones:
                break
            i = random.choice(ones)
            cur &= ~(1 << i)
        
        cur_edges = _count_edges(cur)
        
        # Try to add edges
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            zeros = [i for i in range(64) if not ((cur >> i) & 1)]
            random.shuffle(zeros)
            for i in zeros:
                if time.time() > deadline - 5.0:
                    break
                new_mask = cur | (1 << i)
                if not _has_q3(new_mask):
                    cur = new_mask
                    cur_edges = _count_edges(cur)
                    improved = True
                    if cur_edges > best_edges:
                        best_edges = cur_edges
                        best_mask = cur
        
        # Try simulated-annealing-like: sometimes accept worse states
        for sa_iter in range(500):
            if time.time() > deadline - 5.0:
                break
            i = random.randint(0, 63)
            new_mask = cur ^ (1 << i)
            new_edges = _count_edges(new_mask)
            if not _has_q3(new_mask):
                if new_edges >= cur_edges or random.random() < 0.1:
                    cur = new_mask
                    cur_edges = new_edges
                    if cur_edges > best_edges:
                        best_edges = cur_edges
                        best_mask = cur
            # else: reject (has Q3)
        
        # After SA, try greedy fill again
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            zeros = [i for i in range(64) if not ((cur >> i) & 1)]
            random.shuffle(zeros)
            for i in zeros:
                if time.time() > deadline - 5.0:
                    break
                new_mask = cur | (1 << i)
                if not _has_q3(new_mask):
                    cur = new_mask
                    cur_edges = _count_edges(cur)
                    improved = True
                    if cur_edges > best_edges:
                        best_edges = cur_edges
                        best_mask = cur
    
    # Convert mask to edge list
    edges = []
    for r in range(8):
        for c in range(8):
            if (best_mask >> (r * 8 + c)) & 1:
                edges.append([r, 8 + c])
    return edges
