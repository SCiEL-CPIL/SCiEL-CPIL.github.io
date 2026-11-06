import time
import random
import sys

def construct():
    """
    Finds a large subset of [1, 256] such that for all distinct a, b in the subset,
    (a * b) % (a + b) != 0.
    """
    N = 256
    vertices = list(range(1, N + 1))
    
    # 1. Build conflict graph
    # adj[i] is a set of vertices j such that (i*j) % (i+j) == 0
    adj = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    # Calculate degrees
    degrees = [len(adj[i]) for i in range(N + 1)]
    
    def greedy_construct(order):
        """
        order: list of vertices to consider in this order.
        Returns a set of vertices.
        """
        S = set()
        # We need to quickly check if a vertex has neighbors in S.
        # Since N is small, we can just iterate neighbors.
        for v in order:
            if v in S:
                continue
            # Check if any neighbor of v is in S
            conflict = False
            for u in adj[v]:
                if u in S:
                    conflict = True
                    break
            if not conflict:
                S.add(v)
        return S

    def local_search(S):
        """
        Improves the set S using local moves.
        1. Add any vertex that doesn't conflict with S.
        2. If stuck, try removing one vertex and greedily adding others.
        """
        S = set(S)
        improved = True
        # Limit iterations to avoid infinite loops or excessive time
        max_iter = 100
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Step 1: Try to add any valid vertex
            added = True
            while added:
                added = False
                for v in vertices:
                    if v in S:
                        continue
                    conflict = False
                    for u in adj[v]:
                        if u in S:
                            conflict = True
                            break
                    if not conflict:
                        S.add(v)
                        added = True
                        improved = True
                        break # Restart scan as S changed
            
            if improved:
                continue # Restart the while loop to try adding more
            
            # Step 2: If no vertex can be added, try swapping
            # Try removing each vertex u in S, then greedily add back
            best_S = S
            best_size = len(S)
            
            # We only need to check removals if we hope to gain size.
            # Removing 1, adding >= 2 net gain? Or just larger set.
            for u in list(S):
                S_temp = S - {u}
                # Greedily add vertices to S_temp
                added = True
                while added:
                    added = False
                    for v in vertices:
                        if v in S_temp:
                            continue
                        conflict = False
                        for w in adj[v]:
                            if w in S_temp:
                                conflict = True
                                break
                        if not conflict:
                            S_temp.add(v)
                            added = True
                            break
                
                if len(S_temp) > best_size:
                    best_size = len(S_temp)
                    best_S = S_temp
                    improved = True
                    # We can break here if we found an improvement, 
                    # but checking all might find a better one. 
                    # Given time constraints, taking the first improvement is usually sufficient 
                    # as we will loop again.
                    break 
            
            if improved:
                S = best_S
            else:
                improved = False
        
        return S

    best_set = set()
    best_size = 0
    start_time = time.time()
    deadline = start_time + 85.0 # Leave 5s margin
    
    # Strategy: Multiple random restarts with different orderings
    # Heuristic 1: Sort by degree (ascending)
    # Heuristic 2: Random order
    # Heuristic 3: Random order with bias towards low degree
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Create a randomized ordering
        # We can shuffle the vertices, but prioritize low degree?
        # Let's just shuffle for now, or sort by (random, degree)
        
        # Option A: Pure random shuffle
        order = vertices[:]
        random.shuffle(order)
        
        # Option B: Sort by degree, then randomize ties
        # This is often better for MIS
        # Let's mix: 50% random, 50% degree-sorted
        
        if iterations % 2 == 0:
            # Degree sorted
            order = sorted(vertices, key=lambda x: (degrees[x], random.random()))
        else:
            # Random
            random.shuffle(order)
        
        S = greedy_construct(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        # If we reached a very high size, we might stop early? 
        # Theoretical max is 256. Unlikely to reach.
        # If best_size > 200, we can relax.
        
    return list(best_set)

# To ensure reproducibility in some contexts or just standard practice, 
# we don't strictly need to seed random unless required, but it's good.
# random.seed(42) # Optional

if __name__ == "__main__":
    res = construct()
    print(len(res))
