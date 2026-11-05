import time
import math
import random
import sys
from collections import defaultdict

N = 128
sys.setrecursionlimit(10000)

# Precompute conflicts: for each x, conflicts[x] is a set of (u,v) pairs
# such that {x, u, v} is a forbidden triple (all three pairwise LCMs equal).
conflicts = [set() for _ in range(N + 1)]

def precompute_conflicts():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))

precompute_conflicts()

def can_add(x, current_set):
    """Check if x can be added to current_set without creating a forbidden triple."""
    for u, v in conflicts[x]:
        if u in current_set and v in current_set:
            return False
    return True

def compute_conflict_graph(remaining):
    """
    Build the 2-shadow conflict graph on remaining.
    Two elements u, v in remaining are adjacent if there exists some w (any element, not necessarily in remaining)
    such that {u, v, w} is a forbidden triple.
    Actually, for the independent set formulation: we want no forbidden triple in our final set.
    If we model this as an independent set in a graph G where (u,v) is an edge if {u,v,w} is a forbidden triple for SOME w,
    that's not quite right. An independent set in G means no two elements in our set form a pair that is part of a forbidden triple with ANY third element.
    This is sufficient but not necessary. If u and v are in our set, and there is NO w in our set such that {u,v,w} is forbidden, it's fine.
    The graph edge (u,v) exists if there is ANY w in [1,128] such that {u,v,w} is forbidden.
    If we pick an independent set in this graph, then for any u,v in our set, there is no w in [1,128] such that {u,v,w} is forbidden.
    This implies there is certainly no w in our set.
    So, any independent set in this "pair-conflict" graph is a valid solution.
    This is a stronger constraint than needed (it forbids pairs that *could* form a triple with some external element),
    but it makes the problem a standard MWIS on a graph, which is much easier to bound and solve than the 3-uniform hypergraph case.
    Let's verify: if S is an independent set in G, then for any distinct u,v in S, (u,v) is not an edge in G.
    Edge (u,v) in G means: exists w in [1,128] s.t. lcm(u,v)=lcm(u,w)=lcm(v,w).
    If (u,v) is not an edge, then for all w, it is NOT the case that all three LCMs are equal.
    Therefore, for any w in S, it is not the case that {u,v,w} is a forbidden triple.
    So S contains no forbidden triple.
    This approach is valid. It might be suboptimal (smaller set), but it allows using powerful graph algorithms.
    Given N=128, the graph is sparse. Let's try this.
    """
    # We need to know for each pair (u,v) in remaining, if they are "conflicting" (i.e., part of a forbidden triple with some w).
    # We can precompute this for all pairs in [1,128].
    # pair_conflict[u][v] = True if exists w != u,v such that {u,v,w} is a forbidden triple.
    
    # Let's precompute pair_conflict globally.
    pass

# Global pair conflict matrix
# pair_conflict[i][j] = True if there exists k such that {i,j,k} is a forbidden triple.
pair_conflict = [[False] * (N + 1) for _ in range(N + 1)]

def precompute_pair_conflicts():
    # We can derive this from the triples we found in precompute_conflicts.
    # If (a,b,c) is a forbidden triple, then pair_conflict[a][b] = pair_conflict[a][c] = pair_conflict[b][c] = True.
    for a in range(1, N + 1):
        for (b, c) in conflicts[a]:
            pair_conflict[a][b] = True
            pair_conflict[a][c] = True
            pair_conflict[b][c] = True
            # Symmetry
            pair_conflict[b][a] = True
            pair_conflict[c][a] = True
            pair_conflict[c][b] = True

precompute_pair_conflicts()

def greedy_construct():
    """Greedy construction with random restarts."""
    best_s = set()
    for _ in range(200):
        order = list(range(1, N + 1))
        random.shuffle(order)
        cur = set()
        for x in order:
            if can_add(x, cur):
                cur.add(x)
        if len(cur) > len(best_s):
            best_s = cur
    return best_s

def local_search(s):
    """
    Improve set s by trying to add elements, and if stuck, removing one and trying again.
    """
    s = set(s)
    improved = True
    while improved:
        improved = False
        # Try to add any element
        for x in range(1, N + 1):
            if x not in s and can_add(x, s):
                s.add(x)
                improved = True
                break
        if improved:
            continue
        
        # If no element can be added, try removing one element to allow others to be added
        # We remove an element that has the highest "potential" to unlock additions.
        # Simple heuristic: try removing each element, see if we can add more than 1 back.
        best_removal = None
        best_size = len(s)
        
        # Optimization: only try removing elements that are in the current set
        candidates_to_remove = list(s)
        random.shuffle(candidates_to_remove)
        
        for x in candidates_to_remove:
            s_new = s - {x}
            # Try to add elements to s_new
            added = 0
            for y in range(1, N + 1):
                if y not in s_new and can_add(y, s_new):
                    s_new.add(y)
                    added += 1
                    # Stop if we've already beaten the current size
                    if len(s_new) > best_size:
                        break
            
            if len(s_new) > best_size:
                best_size = len(s_new)
                best_removal = s_new
        
        if best_removal is not None:
            s = best_removal
            improved = True
    return s

def bnb_independent_set(vertices, adjacency, deadline, best_size_ref):
    """
    Branch and Bound for Maximum Independent Set.
    vertices: list of integers (1..N)
    adjacency: dict of set, adj[u] = set of neighbors of u
    deadline: time limit
    best_size_ref: [int] to hold the best size found (modified in place)
    Returns the best set found.
    """
    best_set = set()
    
    # Convert vertices to a list for indexing
    # We will work with a list of candidates.
    
    def rec(candidates, current_set):
        nonlocal best_set, best_size_ref
        
        if time.time() > deadline:
            return
        
        # Upper bound: len(current_set) + len(candidates)
        if len(current_set) + len(candidates) <= best_size_ref[0]:
            return
        
        if not candidates:
            if len(current_set) > best_size_ref[0]:
                best_size_ref[0] = len(current_set)
                best_set = current_set.copy()
            return
        
        # Heuristic: Pick a vertex with high degree in the induced subgraph?
        # Or just pick the first one?
        # Let's pick the vertex with the highest degree in the candidate set.
        # This tends to prune faster.
        
        # Calculate degrees within candidates
        # For speed, just pick candidates[0] if the list is small or random?
        # Let's try picking the one with max degree.
        
        # To avoid O(n^2) per step, we can approximate or just use the first.
        # Given N=128, O(n^2) is fine.
        
        max_deg = -1
        pivot = candidates[0]
        for v in candidates:
            deg = 0
            for u in candidates:
                if u != v and u in adjacency[v]:
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        # Branch on pivot
        # Option 1: Include pivot
        # If we include pivot, we must exclude all its neighbors in candidates.
        new_current = current_set | {pivot}
        new_candidates = [v for v in candidates if v != pivot and v not in adjacency[pivot]]
        
        rec(new_candidates, new_current)
        
        # Option 2: Exclude pivot
        new_candidates = [v for v in candidates if v != pivot]
        rec(new_candidates, current_set)
        
    rec(vertices, set())
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Greedy + Local Search to get a good lower bound
    initial_set = greedy_construct()
    initial_set = local_search(initial_set)
    
    best_size = len(initial_set)
    best_set = initial_set
    
    # 2. Branch and Bound on the Pair-Conflict Graph
    # The pair-conflict graph G has edges (u,v) if {u,v,w} is a forbidden triple for some w.
    # An independent set in G is a valid solution.
    # We want the maximum independent set.
    
    # Build adjacency list for G
    adjacency = defaultdict(set)
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if pair_conflict[u][v]:
                adjacency[u].add(v)
                adjacency[v].add(u)
    
    # Convert best_set to a list of vertices for B&B
    # We run B&B on the full set of vertices [1..N].
    # The B&B will find the maximum independent set.
    # Since the greedy solution is a valid independent set, it provides a good lower bound.
    
    # Note: The B&B might be slow if the graph is dense.
    # Let's check the density.
    # If the graph is sparse, B&B should work well.
    
    vertices = list(range(1, N + 1))
    
    # We use a list to hold the best size found by B&B, initialized with the greedy size.
    # This allows B&B to prune branches that cannot exceed the greedy size.
    bnb_best_size = [best_size]
    
    # Run B&B
    # We need to be careful with recursion depth and time.
    # Let's run it.
    
    bnb_result = bnb_independent_set(vertices, adjacency, deadline, bnb_best_size)
    
    if len(bnb_result) > best_size:
        best_size = len(bnb_result)
        best_set = bnb_result
    
    # If B&B didn't finish or didn't improve, we still have the local search result.
    # We can also try a few more local search rounds if time permits.
    
    if time.time() < deadline - 5.0:
        # Try to improve best_set further with local search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try adding
            for x in range(1, N + 1):
                if x not in best_set and can_add(x, best_set):
                    best_set.add(x)
                    improved = True
                    break
            if improved:
                continue
            # Try removing one and re-optimizing
            # This is expensive, so we do it sparingly.
            # For now, just the add loop is enough for a quick polish.
    
    result = list(best_set)
    if not result:
        result = [1]
        
    return result

# Test locally
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            for k in range(j + 1, len(res)):
                a, b, c = res[i], res[j], res[k]
                if math.lcm(a, b) == math.lcm(a, c) == math.lcm(b, c):
                    valid = False
                    print(f"Invalid triple: {a}, {b}, {c}")
    print(f"Valid: {valid}")
