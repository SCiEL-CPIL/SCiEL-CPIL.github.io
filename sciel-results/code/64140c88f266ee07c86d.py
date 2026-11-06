import time
import random
from itertools import combinations

def construct():
    """
    Construct a K4-free graph on 18 vertices that minimizes the size of the
    largest triangle-free induced subgraph.
    
    Strategy:
    1. Start with a known good structure (e.g., Paley graph P(17) + isolated, or similar).
    2. Use local search (edge flips) to minimize the max triangle-free induced subgraph size.
    3. Use a fast exact check for K4-freeness and a heuristic/exact check for triangle-free sets.
    
    The score is the size of the largest triangle-free induced subgraph.
    We want to minimize this.
    """
    n = 18
    deadline = time.time() + 110  # 110s safety margin
    
    # --- Helper Functions ---
    
    def build_adj(edges):
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    def is_k4_free_adj(adj):
        # Check all 4-sets
        for quad in combinations(range(n), 4):
            i, j, k, l = quad
            if (adj[i][j] and adj[i][k] and adj[i][l] and
                adj[j][k] and adj[j][l] and
                adj[k][l]):
                return False
        return True
    
    def max_tri_free_size(adj, limit=None):
        """
        Find the size of the largest triangle-free induced subgraph.
        Uses a branch and bound approach.
        If limit is provided, returns as soon as a set of size > limit is found? 
        No, we need the exact max to compare. But for speed, we can stop if we know it's small.
        Actually, for local search, we just need to know if it's better than current best.
        Let's implement a function that returns the exact max size.
        """
        best = 0
        # We can use a greedy heuristic to get a lower bound, then branch and bound to confirm.
        # Or just branch and bound with pruning.
        
        # Triangles in the graph? No, triangle-free induced subgraph means the induced subgraph has NO triangles.
        # So we are looking for a set S such that G[S] is triangle-free.
        
        # A set S is triangle-free if for every triple in S, not all 3 edges are present.
        
        # Branch and bound:
        # State: current set S, next vertex to consider, count.
        # Pruning: if count + remaining_vertices <= best, prune.
        # Also, if adding a vertex creates a triangle with two existing vertices in S, we can't add it?
        # No, we can add it, but then S is no longer triangle-free.
        # So we must maintain the invariant that S is triangle-free.
        
        # To check if adding v to S keeps it triangle-free:
        # v must not form a triangle with any pair (u1, u2) in S.
        # i.e., for all u1, u2 in S, not (adj[u1][u2] and adj[u1][v] and adj[u2][v]).
        
        def is_valid_addition(v, S_set, S_list):
            for i in range(len(S_list)):
                u1 = S_list[i]
                if not adj[u1][v]:
                    continue
                for j in range(i + 1, len(S_list)):
                    u2 = S_list[j]
                    if adj[u1][u2] and adj[u2][v]:
                        return False
            return True
        
        def dfs(S_set, S_list, start, count):
            nonlocal best
            if count > best:
                best = count
            
            # Pruning
            if count + (n - start) <= best:
                return
            
            for v in range(start, n):
                if v in S_set:
                    continue
                # Check if adding v keeps S triangle-free
                if is_valid_addition(v, S_set, S_list):
                    S_set.add(v)
                    S_list.append(v)
                    dfs(S_set, S_list, v + 1, count + 1)
                    S_list.pop()
                    S_set.remove(v)
                # Optimization: if we skip v, do we continue? Yes.
            
        dfs(set(), [], 0, 0)
        return best

    def get_paley_edges(q):
        residues = set()
        for x in range(1, q):
            residues.add((x * x) % q)
        edges = []
        for u in range(q):
            for v in range(u + 1, q):
                if (v - u) % q in residues:
                    edges.append([u, v])
        return edges

    # --- Initialization ---
    
    # Candidate 1: Paley(17) + isolated
    edges = get_paley_edges(17)
    adj = build_adj(edges)
    
    # Check K4-free
    if not is_k4_free_adj(adj):
        # Fallback to empty graph or something simple
        edges = []
        adj = build_adj(edges)
    
    # Calculate initial score
    current_score = max_tri_free_size(adj)
    best_score = current_score
    best_edges = edges[:]
    best_adj = adj
    
    # --- Local Search ---
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set(edges)
    
    # Helper to get neighbors of a vertex for faster triangle check?
    # For is_valid_addition, we check pairs in S.
    
    # We will try flipping edges.
    # To speed up max_tri_free_size, we can use a heuristic first?
    # Or just run it. n=18 is small.
    # C(18, 9) ~ 48620. Branch and bound should be fast.
    
    # Let's try to improve.
    # We want to minimize max_tri_free_size.
    # Adding edges makes triangle-free sets smaller (harder to be triangle-free).
    # Removing edges makes triangle-free sets larger (easier to be triangle-free).
    # So we should mostly ADD edges.
    
    # But adding edges might create K4.
    
    # Strategy:
    # 1. Try adding non-existent edges.
    # 2. If K4-free and score decreases, keep.
    # 3. Occasionally try removing edges (less likely to help, but maybe to escape local minima? No, removing increases score usually).
    
    # Let's focus on adding edges.
    
    # To make it faster, we can check K4-freeness locally?
    # Adding edge (u,v) creates a K4 only if there is a pair (w,x) such that w,x are connected to both u and v, and w-x is an edge.
    # i.e., (u,v) + (w,x) + all cross edges.
    # Common neighbors of u and v: N(u) ∩ N(v).
    # If any two common neighbors w,x are connected, then adding (u,v) creates a K4 {u,v,w,x}.
    
    def check_k4_creation(u, v, adj):
        # Common neighbors
        common = []
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                common.append(w)
        # Check if any pair in common is connected
        for i in range(len(common)):
            w1 = common[i]
            for j in range(i + 1, len(common)):
                w2 = common[j]
                if adj[w1][w2]:
                    return True
        return False
    
    # Local Search Loop
    random.seed(42)
    
    # To speed up, we can precompute common neighbors?
    # Or just compute on the fly. n=18 is small.
    
    steps = 0
    no_improve = 0
    max_no_improve = 500  # Restart or stop if stuck
    
    while time.time() < deadline:
        # Try a random edge addition
        # Pick a random non-edge
        # To be efficient, maintain a list of non-edges?
        # Or just pick random u,v and check if not edge.
        
        u, v = random.sample(range(n), 2)
        if adj[u][v]:
            continue
        
        # Check K4
        if check_k4_creation(u, v, adj):
            continue
        
        # Tentatively add edge
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((u, v))
        
        # Calculate new score
        # This is the expensive part.
        # Can we update max_tri_free_size incrementally?
        # Adding an edge (u,v) can only decrease the max triangle-free set size.
        # Specifically, any triangle-free set containing both u and v is no longer valid (if it was valid before, it must not have had a triangle, but now {u,v} is an edge... wait.
        # A set S is triangle-free if no triangle is fully inside S.
        # Adding edge (u,v) creates a triangle in S if S contains u, v, and some w connected to both.
        # So sets containing u and v might become invalid.
        # This is complex to update incrementally.
        
        new_score = max_tri_free_size(adj)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = [ [u, v] for u, v in edge_set ]
            best_adj = [row[:] for row in adj]
            no_improve = 0
            # Keep the edge
        else:
            # Revert edge
            adj[u][v] = False
            adj[v][u] = False
            edge_set.remove((u, v))
            no_improve += 1
            
            if no_improve > max_no_improve:
                # Restart with a different seed or randomization?
                # Or just break if we are stuck for too long
                # Let's try a few random restarts if stuck
                # For now, just continue, maybe shuffle order?
                # We can also try removing edges if we are stuck?
                # But removing edges usually increases score.
                # Let's try a random restart of the local search phase with a different initial graph?
                # No, let's just keep trying random edges.
                
                # To avoid infinite loop in same local minimum, we can accept neutral moves?
                # If new_score == best_score, we might accept it to move around?
                # Let's try: if new_score <= best_score, accept.
                
                # Re-do the check with <=
                # Actually, I already reverted.
                # Let's modify the logic:
                # If new_score < best_score: Accept
                # If new_score == best_score: Accept with prob 0.5?
                # For now, let's just restart the no_improve counter if we find a neutral move?
                # No, let's stick to strict improvement for simplicity, but if stuck, try random restart.
                
                # Let's implement a simple restart:
                # Generate a new random K4-free graph?
                # Or just keep going.
                
                pass
        steps += 1
        
        # Safety check
        if steps % 100 == 0:
            # Do nothing, just check time
            pass

    # Return the best edges found
    # Ensure we return a list of lists [u, v]
    return best_edges

# Run a quick test
if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
    # Check K4 free
    n = 18
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    for quad in combinations(range(n), 4):
        i, j, k, l = quad
        if (adj[i][j] and adj[i][k] and adj[i][l] and
            adj[j][k] and adj[j][l] and
            adj[k][l]):
            print("K4 found!")
            break
    else:
        print("K4-free")
    
    # Check max tri free
    # Re-use logic
    def max_tri_free_size(adj, limit=None):
        best = 0
        def is_valid_addition(v, S_set, S_list):
            for i in range(len(S_list)):
                u1 = S_list[i]
                if not adj[u1][v]:
                    continue
                for j in range(i + 1, len(S_list)):
                    u2 = S_list[j]
                    if adj[u1][u2] and adj[u2][v]:
                        return False
            return True
        def dfs(S_set, S_list, start, count):
            nonlocal best
            if count > best:
                best = count
            if count + (n - start) <= best:
                return
            for v in range(start, n):
                if v in S_set:
                    continue
                if is_valid_addition(v, S_set, S_list):
                    S_set.add(v)
                    S_list.append(v)
                    dfs(S_set, S_list, v + 1, count + 1)
                    S_list.pop()
                    S_set.remove(v)
        dfs(set(), [], 0, 0)
        return best
        
    score = max_tri_free_size(adj)
    print(f"Score: {score}")
