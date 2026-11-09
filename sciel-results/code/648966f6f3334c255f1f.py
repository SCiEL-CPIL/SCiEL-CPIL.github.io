"""Dense Q3-free graphs on 16 vertices: local search / simulated annealing."""
import time
import random
import itertools
import sys

# Deadline guard
_DEADLINE = 150.0  # seconds wall-clock limit
_START = None


def _elapsed():
    if _START is None:
        return 0.0
    return time.time() - _START


def _time_left():
    return _DEADLINE - _elapsed()


def _has_q3(edges_set, n=16):
    """Check if the graph (edges_set is a set of frozenset({u,v}) or tuple (min,max)) contains a Q3 subgraph."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for e in edges_set:
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    
    # Q3 has 8 vertices and 12 edges
    # Enumerate all 8-subsets: C(16,8) = 12870
    # For each, check if induced subgraph is isomorphic to Q3
    # Q3: 8 vertices, each degree 3, 12 edges total
    # Actually we need a subgraph isomorphic to Q3, not necessarily induced.
    # But Q3 is a 3-regular bipartite graph on 8 vertices with 12 edges.
    # A subgraph isomorphic to Q3 means there exist 8 distinct vertices and 12 edges among them forming Q3.
    
    # Faster approach: Q3 is bipartite with parts of size 4.
    # Enumerate pairs of 4-sets (A, B) where A ∩ B = ∅, |A| = |B| = 4
    # Then check if all 12 edges of a perfect matching structure exist... 
    # Actually Q3 as a bipartite graph: vertices are {0,1}^3, edges between x and y if Hamming distance 1.
    # The bipartition is by parity of sum of bits.
    
    # Let's use a different approach: enumerate all 8-vertex subsets and check if they contain a Q3.
    # C(16,8) = 12870, for each we check 12 edges. That's about 154k edge checks per call.
    # But we call this many times... need to be smarter.
    
    # Alternative: precompute all Q3 embeddings (8-vertex subsets with the 12 specific edges).
    # Number of ways to embed Q3 in K16: choose 8 vertices (12870), then number of isomorphisms of Q3 onto those 8 vertices.
    # Aut(Q3) has order 48 (hyperoctahedral group B3). So for each 8-set, there are 8!/48 = 2880 distinct edge sets that form a Q3.
    # Total Q3 subgraphs in K16: 12870 * 2880 ≈ 37 million. Too many to precompute all.
    
    # Better: for each 8-vertex subset, check if the induced subgraph on those 8 vertices contains a Q3 as a subgraph.
    # The induced subgraph has at most C(8,2)=28 edges. We need to check if it contains 12 specific edges forming Q3.
    # This is still expensive per 8-set.
    
    # Let me use a smarter check: Q3 is 3-regular. So all 8 vertices must have degree >= 3 within the 8-set in the subgraph.
    # First filter: all 8 vertices have degree >= 3 in the induced subgraph.
    
    for vertices in itertools.combinations(range(n), 8):
        vset = set(vertices)
        # Check degrees in induced subgraph
        ok = True
        for v in vertices:
            deg = 0
            for w in vertices:
                if v != w and w in adj[v]:
                    deg += 1
            if deg < 3:
                ok = False
                break
        if not ok:
            continue
        # Now check if this 8-vertex induced subgraph contains a Q3 subgraph
        # Get the edges among these 8 vertices
        sub_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                if vertices[j] in adj[vertices[i]]:
                    sub_edges.append((i, j))
        # We need to check if there's a subset of 12 edges forming Q3
        # Q3 structure: 8 vertices, 12 edges, 3-regular, bipartite
        # The bipartition: in Q3, vertices are {0,1}^3, partitioned by parity.
        # For the 8 vertices labeled 0-7, we need to find a bipartition (4,4) such that
        # all 12 edges go across, and each vertex has degree 3.
        
        # There are C(8,4)/2 = 35 possible bipartitions (up to complement).
        # For each bipartition (A, B), check if all edges in sub_edges go from A to B,
        # and the bipartite graph is 3-regular.
        
        # Actually, we just need to check: does there exist a labeling of the 8 vertices as {0,1}^3
        # such that the 12 edges of Q3 are all present in sub_edges?
        
        # Simpler: iterate over all 35 bipartitions, check if the bipartite graph is exactly 3-regular
        # with 12 edges.
        
        # Precompute all 35 bipartitions of 8 elements into two 4-sets
        # We can generate them on the fly
        
        found = False
        # Generate all ways to choose 4 of 8 for set A (the rest is B)
        # To avoid duplicates, fix vertex 0 in A
        for A in itertools.combinations(range(1, 8), 3):
            A = frozenset([0] + list(A))
            B = frozenset(set(range(8)) - A)
            # Check bipartite: all edges cross
            bip_ok = True
            for (i, j) in sub_edges:
                if i in A and j in A:
                    bip_ok = False
                    break
                if i in B and j in B:
                    bip_ok = False
                    break
            if not bip_ok:
                continue
            # Check 3-regular
            deg = [0]*8
            for (i, j) in sub_edges:
                deg[i] += 1
                deg[j] += 1
            if all(d == 3 for d in deg) and len(sub_edges) == 12:
                found = True
                break
        if found:
            return True
    return False


def _has_q3_fast(edges_set, n=16):
    """Faster Q3 check using the bipartite structure."""
    adj = [set() for _ in range(n)]
    for e in edges_set:
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    
    # Q3 is bipartite with parts of size 4.
    # Enumerate all pairs of disjoint 4-sets (A, B)
    # For each pair, check if the bipartite subgraph between A and B is 3-regular with 12 edges.
    # Number of ways: C(16,4) * C(12,4) / 2 = 1820 * 495 / 2 = 450450. Too many.
    
    # Better: for each 8-vertex subset, check as before but with early termination.
    # C(16,8) = 12870. For each, we do limited work.
    
    for vertices in itertools.combinations(range(n), 8):
        vset = set(vertices)
        # Quick degree check
        for v in vertices:
            deg = sum(1 for w in vertices if w != v and w in adj[v])
            if deg < 3:
                break
        else:
            # All degrees >= 3, check for Q3
            sub_edges = []
            for i in range(8):
                for j in range(i+1, 8):
                    if vertices[j] in adj[vertices[i]]:
                        sub_edges.append((i, j))
            if len(sub_edges) < 12:
                continue
            # Check bipartite 3-regular
            found = False
            for A in itertools.combinations(range(1, 8), 3):
                A = frozenset([0] + list(A))
                B = frozenset(set(range(8)) - A)
                bip_ok = True
                for (i, j) in sub_edges:
                    if (i in A and j in A) or (i in B and j in B):
                        bip_ok = False
                        break
                if not bip_ok:
                    continue
                deg = [0]*8
                for (i, j) in sub_edges:
                    deg[i] += 1
                    deg[j] += 1
                if all(d == 3 for d in deg):
                    found = True
                    break
            if found:
                return True
    return False


def _add_edge_creates_q3(current_edges_set, u, v, n=16):
    """Check if adding edge (u,v) to current_edges_set creates a Q3."""
    new_edges = current_edges_set | {frozenset((u, v))}
    return _has_q3_fast(new_edges, n)


def construct():
    global _START
    _START = time.time()
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_possible = len(all_edges)  # 120
    
    # Start with K3,13: vertices 0,1,2 connected to 3,4,...,15
    initial_edges = set()
    for a in range(3):
        for b in range(3, 16):
            initial_edges.add(frozenset((a, b)))
    
    best_edges = set(initial_edges)
    best_count = len(best_edges)  # 39
    
    # Try to add more edges using local search
    # First, try greedy addition
    remaining = [e for e in all_edges if frozenset(e) not in best_edges]
    random.shuffle(remaining)
    
    for e in remaining:
        if _time_left() < 20:
            break
        if _add_edge_creates_q3(best_edges, e[0], e[1], n):
            continue
        best_edges.add(frozenset(e))
        best_count += 1
    
    # Now try simulated annealing / local search to improve
    # State: set of edges. Move: add or remove an edge.
    # Objective: maximize number of edges subject to Q3-free constraint.
    
    current_edges = set(best_edges)
    current_count = len(current_edges)
    
    # Precompute which edges are in current and which are not
    in_graph = set(current_edges)
    not_in_graph = [e for e in all_edges if frozenset(e) not in in_graph]
    
    T = 10.0  # temperature
    T_min = 0.01
    cooling = 0.9995
    
    random.shuffle(not_in_graph)
    
    step = 0
    while _time_left() > 5:
        step += 1
        if T <= T_min:
            T = T_min
        
        # Choose a move: try to add a random edge from not_in_graph, or remove a random edge from in_graph
        if len(not_in_graph) > 0 and random.random() < 0.8:
            # Try to add
            idx = random.randint(0, len(not_in_graph)-1)
            e = not_in_graph[idx]
            new_edges = in_graph | {frozenset(e)}
            if _has_q3_fast(new_edges, n):
                # Reject: would create Q3
                # With SA, we might accept with some probability, but it's invalid
                # So we just reject
                continue
            else:
                # Accept
                in_graph = new_edges
                not_in_graph.pop(idx)
                current_count += 1
                if current_count > best_count:
                    best_edges = set(in_graph)
                    best_count = current_count
        else:
            # Try to remove
            if len(in_graph) == 0:
                continue
            e = random.choice(list(in_graph))
            new_edges = in_graph - {e}
            in_graph = new_edges
            not_in_graph.append(e)
            current_count -= 1
        
        # Cool down
        T *= cooling
    
    # Final greedy attempt from best
    if _time_left() > 10:
        best_edges = set(best_edges)
        remaining2 = [e for e in all_edges if frozenset(e) not in best_edges]
        random.shuffle(remaining2)
        for e in remaining2:
            if _time_left() < 5:
                break
            if not _add_edge_creates_q3(best_edges, e[0], e[1], n):
                best_edges.add(frozenset(e))
    
    # Return as list of [u, v] pairs
    result = [[int(min(e)), int(max(e))] for e in best_edges]
    return result


# Quick test
if __name__ == "__main__":
    _START = time.time()
    result = construct()
    print(f"Edges: {len(result)}")
    # Verify
    es = set(frozenset((e[0], e[1])) for e in result)
    print(f"Q3-free: {not _has_q3_fast(es)}")
