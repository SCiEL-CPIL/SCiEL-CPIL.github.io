"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_rainbow_4cycles(edges, deadline):
    """
    Check if every 4-cycle in the graph formed by edges has 4 distinct colours.
    edges: list of [u, v, c]
    Returns True if valid, False otherwise.
    """
    n = 20
    # adj[u][v] = colour of edge (u,v)
    adj = [[None] * n for _ in range(n)]
    for u, v, c in edges:
        adj[u][v] = c
        adj[v][u] = c

    # A 4-cycle is a sequence a-b-c-d-a with a<b<c<d? No, just distinct vertices.
    # To avoid duplicates, we can iterate over pairs of disjoint edges (a,b) and (c,d) 
    # where {a,b} and {c,d} are disjoint and a<c, b<d? 
    # Actually, a 4-cycle is defined by 4 distinct vertices a,b,c,d such that 
    # edges (a,b), (b,c), (c,d), (d,a) exist.
    # To check all 4-cycles without duplication, we can iterate over all 4-subsets 
    # of vertices and check if they form a 4-cycle in any of the 3 possible pairings?
    # No, a 4-cycle on vertices {a,b,c,d} exists if the induced subgraph contains a cycle of length 4.
    # The possible 4-cycles on 4 vertices are:
    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a.
    # But these are just permutations.
    # A simpler way: iterate over all pairs of vertices (a, b) and (c, d) such that 
    # a < b, c < d, and {a,b} ∩ {c,d} = ∅.
    # If edges (a,b), (b,c), (c,d), (d,a) exist, we have a cycle? No, that's not necessarily a simple cycle.
    # The standard way to enumerate 4-cycles:
    # For each pair of non-adjacent vertices (a, c), find common neighbors b, d.
    # If b and d are distinct, then a-b-c-d-a is a 4-cycle.
    # We need to check if colours of (a,b), (b,c), (c,d), (d,a) are distinct.
    
    # To be efficient and avoid checking the same cycle multiple times, 
    # we can enforce an ordering on the vertices.
    # Let's iterate over all 4-combinations of vertices {a, b, c, d} with a < b < c < d.
    # There are 3 possible 4-cycles on these 4 vertices:
    # 1. a-b-c-d-a (edges: (a,b), (b,c), (c,d), (d,a))
    # 2. a-b-d-c-a (edges: (a,b), (b,d), (d,c), (c,a))
    # 3. a-c-b-d-a (edges: (a,c), (c,b), (b,d), (d,a))
    # If any of these forms a valid cycle (all 4 edges exist) and the colours are not distinct, return False.
    
    for a, b, c, d in combinations(range(n), 4):
        if time.time() > deadline:
            return False  # Timeout, assume invalid to be safe? Or return True? 
                          # Better to return False if we can't finish, but the caller should handle deadline.
                          # Actually, if we hit deadline, we should probably return True to avoid false negative 
                          # if the graph is valid but we ran out of time. But the task says "checked over all 4-cycles".
                          # If we don't finish checking, we can't be sure. 
                          # However, the deadline guard is in the main function. 
                          # Let's assume this function is called only if there's enough time.
        
        # Check cycle 1: a-b-c-d-a
        if (adj[a][b] is not None and adj[b][c] is not None and adj[c][d] is not None and adj[d][a] is not None):
            colors = [adj[a][b], adj[b][c], adj[c][d], adj[d][a]]
            if len(set(colors)) != 4:
                return False
        
        # Check cycle 2: a-b-d-c-a
        if (adj[a][b] is not None and adj[b][d] is not None and adj[d][c] is not None and adj[c][a] is not None):
            colors = [adj[a][b], adj[b][d], adj[d][c], adj[c][a]]
            if len(set(colors)) != 4:
                return False
        
        # Check cycle 3: a-c-b-d-a
        if (adj[a][c] is not None and adj[c][b] is not None and adj[b][d] is not None and adj[d][a] is not None):
            colors = [adj[a][c], adj[c][b], adj[b][d], adj[d][a]]
            if len(set(colors)) != 4:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin under 120s limit

    # Strategy:
    # 1. Start with the complete bipartite graph K(10,10) with the proven valid colouring.
    # 2. Try to add edges within the partitions (U-U or V-V) or between U and V that are not in K(10,10)?
    #    K(10,10) already has all U-V edges. So we can only add U-U edges or V-V edges.
    # 3. Adding an intra-partition edge creates 4-cycles involving that edge and two cross edges.
    #    For example, if we add edge (u1, u2) in U, and we have v1, v2 in V, then u1-v1-u2-v2-u1 is a 4-cycle?
    #    No, u1-v1-u2-v2-u1 requires edges (u1,v1), (v1,u2), (u2,v2), (v2,u1).
    #    This is a 4-cycle in the bipartite graph.
    #    Wait, if we add an edge (u1, u2) within U, does it create new 4-cycles?
    #    A 4-cycle must have 4 distinct vertices.
    #    If we have edge (u1, u2) in U, and we pick v1, v2 in V, the cycle u1-u2-v1-v2-u1 requires:
    #    (u1,u2), (u2,v1), (v1,v2), (v2,u1).
    #    But (v1,v2) is an intra-V edge, which we don't have yet.
    #    So adding a single intra-U edge doesn't immediately create a 4-cycle unless there's an intra-V edge.
    #    However, it creates 4-cycles like u1-u2-v1-v2-u1? No, that's not a cycle in the graph unless (v1,v2) exists.
    #    What about u1-v1-u2-v2-u1? This is a 4-cycle in the bipartite graph, which already exists and is valid.
    #    The new edge (u1,u2) is not part of this cycle.
    #    So adding intra-partition edges might be safe as long as we don't create 4-cycles involving two intra-partition edges.
    #    A 4-cycle involving two intra-U edges would require two vertices in U and two in V, but the cycle would be u1-u2-v1-v2-u1? No, that needs (v1,v2).
    #    Or u1-v1-u2-v2-u1? This uses no intra-U edges.
    #    Actually, any 4-cycle in a bipartite graph uses only cross edges.
    #    If we add intra-partition edges, we create non-bipartite cycles.
    #    A 4-cycle could be u1-u2-v1-v2-u1 if (v1,v2) exists.
    #    Or u1-v1-v2-u2-u1 if (v1,v2) exists.
    #    So, if we add intra-U edges and intra-V edges, we must be careful.
    #    If we only add intra-U edges, can we form a 4-cycle?
    #    A 4-cycle needs 4 edges. If it uses one intra-U edge (u1,u2), the other 3 edges must connect u1,u2 to other vertices.
    #    Since the graph is almost bipartite, the other edges must be cross edges.
    #    Path: u1 - u2 - v1 - v2 - u1? No, (v1,v2) is missing.
    #    Path: u1 - u2 - v1 - u1? That's a triangle, not a 4-cycle.
    #    Path: u1 - u2 - v1 - v2 - u1? Requires (v1,v2).
    #    Path: u1 - v1 - u2 - v2 - u1? This is a bipartite 4-cycle, already exists.
    #    So, adding intra-U edges alone does NOT create any new 4-cycles, because any 4-cycle involving an intra-U edge would need another intra-U or intra-V edge to close the loop with 4 distinct vertices.
    #    Wait, is that true?
    #    Let's check: vertices u1, u2 in U, v1, v2 in V.
    #    Edges: (u1,u2) [new], (u2,v1), (v1,v2) [missing], (v2,u1).
    #    Without (v1,v2), no cycle.
    #    What if we have u1, u2, u3 in U?
    #    Cycle u1-u2-v1-u3-v1-u1? No, v1 repeats.
    #    So, it seems we can add ALL intra-U edges and ALL intra-V edges?
    #    Let's test: If we add all intra-U edges, do we create 4-cycles?
    #    Consider u1, u2 in U and v1, v2 in V.
    #    Is there a 4-cycle using (u1,u2)?
    #    Possible cycles:
    #    1. u1-u2-v1-v2-u1: Needs (v1,v2). If we don't add intra-V edges, this doesn't exist.
    #    2. u1-v1-u2-v2-u1: This is a bipartite cycle, already exists.
    #    So, if we only add intra-U edges, we are safe.
    #    Similarly, if we only add intra-V edges, we are safe.
    #    But if we add BOTH intra-U and intra-V edges, then u1-u2-v1-v2-u1 becomes a valid 4-cycle.
    #    We need to check if its colours are distinct.
    #    
    #    Let's try to add all intra-U edges first.
    #    Colours for intra-U edges: We have 10 U vertices, so 45 edges.
    #    We need to assign colours from 0..19.
    #    We must ensure that no 4-cycle has repeated colours.
    #    Since adding intra-U edges doesn't create new 4-cycles (as argued above, assuming no intra-V edges), we can assign arbitrary colours?
    #    Wait, the existing bipartite 4-cycles must remain rainbow. They don't involve intra-U edges.
    #    So the intra-U edges don't affect the validity of the bipartite 4-cycles.
    #    And no new 4-cycles are created.
    #    So we can add all intra-U edges with ANY colours?
    #    Let's double check.
    #    Is it possible to have a 4-cycle like u1-u2-u3-v1-u1? No, u1-u2-u3 is a path, u3-v1 is cross, v1-u1 is cross. That's a 4-cycle?
    #    Vertices: u1, u2, u3, v1.
    #    Edges: (u1,u2), (u2,u3), (u3,v1), (v1,u1).
    #    This is a 4-cycle!
    #    I missed this. A 4-cycle can have 3 vertices in one partition and 1 in the other.
    #    So adding intra-U edges DOES create new 4-cycles.
    #    We must check these new 4-cycles.
    #    
    #    Strategy:
    #    1. Start with K(10,10) with the valid colouring.
    #    2. Add intra-U edges one by one, checking validity.
    #    3. Add intra-V edges one by one, checking validity.
    #    4. If an edge causes a violation, skip it or try a different colour.
    
    U = list(range(0, 10))
    V = list(range(10, 20))
    
    edges = []
    for u in U:
        for v in V:
            v_prime = v - 10
            c = (u + 10 * v_prime) % 20
            edges.append([u, v, c])
    
    # Try to add intra-U edges
    for i in range(len(U)):
        for j in range(i+1, len(U)):
            if time.time() > deadline - 10.0:
                break
            u1, u2 = U[i], U[j]
            # Try colours 0..19
            added = False
            for c in range(20):
                new_edges = edges + [[u1, u2, c]]
                if _check_rainbow_4cycles(new_edges, deadline):
                    edges = new_edges
                    added = True
                    break
            if not added:
                pass # Skip this edge
            
        if time.time() > deadline - 10.0:
            break
    
    # Try to add intra-V edges
    for i in range(len(V)):
        for j in range(i+1, len(V)):
            if time.time() > deadline - 10.0:
                break
            v1, v2 = V[i], V[j]
            added = False
            for c in range(20):
                new_edges = edges + [[v1, v2, c]]
                if _check_rainbow_4cycles(new_edges, deadline):
                    edges = new_edges
                    added = True
                    break
            if not added:
                pass

    return edges

# EVOLVE-BLOCK-END
