"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # We need minimum edges in a 3-uniform hypergraph on 9 vertices
    # that is NOT 2-colorable (chromatic number >= 3)
    # and has a valid 3-coloring where no edge is monochromatic.
    
    # Strategy: search for small edge sets and check 2-colorability.
    # A hypergraph is 2-colorable if there exists a 2-coloring with no monochromatic edge.
    
    def is_2_colorable(edges, n=9):
        """Check if the hypergraph is 2-colorable."""
        # Try all 2^9 colorings (512 possibilities)
        for mask in range(1 << n):
            valid = True
            for edge in edges:
                colors = [(mask >> v) & 1 for v in edge]
                if len(set(colors)) == 1:
                    valid = False
                    break
            if valid:
                return True
        return False
    
    def has_valid_3_coloring(edges, colors, n=9):
        """Check that the given 3-coloring makes no edge monochromatic."""
        for edge in edges:
            if colors[edge[0]] == colors[edge[1]] == colors[edge[2]]:
                return False
        return True
    
    # We'll search incrementally by number of edges
    # Start from 1 edge and go up
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(9), 3))
    
    # For each number of edges k, check if there exists a set of k edges
    # that is not 2-colorable and has a valid 3-coloring.
    
    # We need to find the minimum k.
    # Let's try k=1, 2, 3, ... until we find one.
    
    # For small k, we can enumerate combinations of edges.
    # C(84, k) for k=1: 84, k=2: 3486, k=3: 95284, k=4: 2.3M, k=5: 51M...
    
    # Let's be smart: for each k, iterate through combinations and check.
    
    best_result = None
    
    for k in range(1, 8):  # try up to 7 edges
        if time.time() > deadline - 5:
            break
        
        if k == 1:
            # Any single edge is 2-colorable, skip
            continue
        elif k == 2:
            # Two edges: can they be non-2-colorable?
            # Each edge individually is 2-colorable. With 2 edges, 
            # we need both to be non-monochromatic simultaneously.
            # This is almost always possible. Let's check.
            found = False
            for i in range(len(all_triples)):
                for j in range(i+1, len(all_triples)):
                    if time.time() > deadline - 5:
                        break
                    edges = [all_triples[i], all_triples[j]]
                    if not is_2_colorable(edges):
                        # Check 3-coloring
                        # We need a 3-coloring where no edge is monochromatic
                        # Try to find one
                        colors = [0]*9
                        # Simple: color vertices 0,1,2 with colors 0,1,2
                        # and rest with 0
                        colors = [0,0,1,1,2,0,0,0,0]
                        if has_valid_3_coloring(edges, colors):
                            best_result = {'edges': [list(e) for e in edges], 'colors': colors}
                            found = True
                            break
                if found:
                    break
            if found:
                return best_result
        elif k == 3:
            # C(84,3) = 95284 combinations, each check is 512 * 3 ops ~ 1536
            # Total: ~146M ops, might be tight. Let's try with early termination.
            found = False
            count = 0
            for i in range(len(all_triples)):
                if time.time() > deadline - 10:
                    break
                for j in range(i+1, len(all_triples)):
                    if time.time() > deadline - 10:
                        break
                    for l in range(j+1, len(all_triples)):
                        count += 1
                        if time.time() > deadline - 10:
                            break
                        edges = [all_triples[i], all_triples[j], all_triples[l]]
                        if not is_2_colorable(edges):
                            colors = [0,0,1,1,2,0,0,0,0]
                            if has_valid_3_coloring(edges, colors):
                                best_result = {'edges': [list(e) for e in edges], 'colors': colors}
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if found:
                return best_result
        elif k == 4:
            # C(84,4) = 2.3M, each check ~512*4=2048 ops, total ~4.7B - too slow
            # Need smarter approach. Use random sampling or smarter search.
            import random
            found = False
            random.seed(42)
            attempts = 0
            max_attempts = 200000
            while attempts < max_attempts and not found:
                if time.time() > deadline - 10:
                    break
                attempts += 1
                edges = [all_triples[random.randint(0, len(all_triples)-1)] for _ in range(k)]
                # Remove duplicates
                edges = list(set(tuple(e) for e in edges))
                if len(edges) < k:
                    continue
                edges = [list(e) for e in edges]
                if not is_2_colorable(edges):
                    colors = [0,0,1,1,2,0,0,0,0]
                    if has_valid_3_coloring(edges, colors):
                        best_result = {'edges': edges, 'colors': colors}
                        found = True
            if found:
                return best_result
        elif k >= 5:
            import random
            found = False
            random.seed(123)
            attempts = 0
            max_attempts = 500000
            while attempts < max_attempts and not found:
                if time.time() > deadline - 10:
                    break
                attempts += 1
                edges = [all_triples[random.randint(0, len(all_triples)-1)] for _ in range(k)]
                edges = list(set(tuple(e) for e in edges))
                if len(edges) < k:
                    continue
                edges = [list(e) for e in edges]
                if not is_2_colorable(edges):
                    colors = [0,0,1,1,2,0,0,0,0]
                    if has_valid_3_coloring(edges, colors):
                        best_result = {'edges': edges, 'colors': colors}
                        found = True
            if found:
                return best_result
    
    # Fallback: use the original answer
    return {'edges': [list(e) for e in itertools.combinations(range(5), 3)], 'colors': [0,0,1,1,2,0,0,0,0]}

# EVOLVE-BLOCK-END
