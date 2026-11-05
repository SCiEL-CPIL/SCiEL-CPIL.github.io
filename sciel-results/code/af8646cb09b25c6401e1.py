import time
import random
from itertools import combinations

def construct():
    """
    Return a 5-uniform hypergraph with no proper 2-colouring, minimising edges.
    
    Strategy:
      1. Use a fast bitmask-based checker for n <= 20 (exploits small n).
      2. Multi-start random search over n=9,10,11 with varied initial edge counts.
      3. Aggressive greedy pruning with random restarts to escape local minima.
      4. Keep the best (fewest edges) result found within the time budget.
      5. Safety: always return a valid non-2-colorable hypergraph.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    best_edges = None
    best_n = 9

    # Helper: Fast check for non-2-colorability using bitmasks
    # Only efficient for small n (<= 20 or so)
    def is_non_2colorable_fast(edges, n):
        """
        Returns True if no 2-colouring exists where every edge has both colours.
        i.e., For every mask (subset of red vertices), there is an edge e 
        such that |e & mask| == 0 or |e & mask| == 5.
        """
        if not edges:
            return False

        # Precompute edge masks
        edge_masks = []
        for e in edges:
            m = 0
            for v in e:
                m |= (1 << v)
            edge_masks.append(m)
        
        # Iterate over all possible colorings (masks)
        # mask bit i is 1 if vertex i is red
        # Optimization: skip mask 0 and mask (1<<n)-1 as they are trivially invalid
        # but checking all is fine for n<=11 (2^11=2048)
        for mask in range(1 << n):
            valid_coloring = True
            for em in edge_masks:
                # Count red vertices in this edge
                # bin(x).count('1') is fast for small x
                if bin(em & mask).count('1') in (0, 5):
                    valid_coloring = False
                    break
            if valid_coloring:
                # Found a coloring where all edges have both colors
                return False
        return True

    # Helper: Greedy pruning with random restarts
    def prune_edges(edges, n, time_limit):
        """
        Try to remove edges while preserving non-2-colorability.
        Uses multiple passes with shuffled orders to find a better local minimum.
        Returns the pruned list of edges.
        """
        current_edges = list(edges)
        
        # We will do several passes. In each pass, we shuffle and try to remove edges.
        # We stop if we hit the time limit or if we can't remove any more edges in a full pass.
        
        # First, do a quick single pass to get a baseline
        current_edges = _prune_pass(current_edges, n, time_limit)
        
        # Then, try to improve with random restarts/shuffles
        # Limit restarts to avoid spending too much time if stuck
        max_restarts = 5
        for _ in range(max_restarts):
            if time.time() > time_limit:
                break
            if len(current_edges) <= 29: # Theoretical lower bound is 29, so if we are at 30, try to get to 29
                break
            
            # Shuffle the current edges to try removing different ones
            shuffled = current_edges[:]
            random.shuffle(shuffled)
            
            new_edges = _prune_pass(shuffled, n, time_limit)
            
            if len(new_edges) < len(current_edges):
                current_edges = new_edges
            else:
                # If we couldn't improve, we might be stuck in a local minimum.
                # We could try a "kick" (remove a random edge, check, if fail put back)
                # But for now, just stop if no progress in a pass.
                break
                
        return current_edges

    def _prune_pass(edges, n, time_limit):
        """
        Single pass of greedy pruning.
        """
        current_edges = list(edges)
        i = 0
        while i < len(current_edges):
            if time.time() > time_limit:
                break
            
            # Try removing current_edges[i]
            candidate = current_edges[:i] + current_edges[i+1:]
            
            # Check if candidate is still non-2-colorable
            if is_non_2colorable_fast(candidate, n):
                current_edges = candidate
                # Do not increment i, check the next edge in the new list at the same index
            else:
                i += 1
        return current_edges

    # Main search loop
    # We try n=9 and n=10. n=9 has fewer vertices, so 2^n checks are faster,
    # but fewer total possible edges (126). n=10 has 252 possible edges.
    # Known best is 51 edges.
    
    # Pre-compute all possible edges for each n to avoid re-computing in loop
    all_possible_edges = {}
    for n in [9, 10, 11]:
        all_possible_edges[n] = list(combinations(range(n), 5))
        
    for n in [9, 10, 11]:
        if time.time() > deadline - 10.0:
            break
            
        # Number of all possible edges
        total_edges = len(all_possible_edges[n])
        
        # Try several random instances
        # We start with a reasonable number of edges, say 50-60
        # If we find a non-2-colorable one, we prune it.
        
        # Increase attempts for smaller n as checks are faster
        attempts = 2000 if n == 9 else 1000
        
        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break
                
            # Choose a target number of edges for initialization
            # We want to find small hypergraphs. 
            # If we start too small, it's likely colorable.
            # If we start too large, it's likely non-2-colorable but pruning takes time.
            # Let's start around 45-60.
            k_init = random.randint(45, 65)
            
            # Generate random distinct edges
            all_possible = all_possible_edges[n][:]
            random.shuffle(all_possible)
            initial_edges = [list(e) for e in all_possible[:k_init]]
            
            # Check if initial is non-2-colorable
            if is_non_2colorable_fast(initial_edges, n):
                # It is non-2-colorable. Try to prune.
                # Set a sub-deadline for pruning to allow restarts
                prune_deadline = time.time() + 2.0
                if time.time() + 2.0 > deadline:
                    prune_deadline = deadline - 1.0
                    
                pruned = prune_edges(initial_edges, n, prune_deadline)
                
                if best_edges is None or len(pruned) < len(best_edges):
                    best_edges = pruned
                    best_n = n
                    
                    # If we found a very good solution, we can keep searching for better
                    # But if we hit the known lower bound or close to it, it's great.
                    # The known lower bound is 29.
                    if len(pruned) <= 30:
                        # Unlikely, but if we find it, we can stop early or keep trying
                        pass
                        
    # If no solution found (should not happen as random dense graphs are non-2-colorable),
    # fall back to a known construction or the complete graph on 9 vertices.
    if best_edges is None:
        # Fallback: Complete 5-graph on 9 vertices
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
