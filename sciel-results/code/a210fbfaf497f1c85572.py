# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Known 3-chromatic triple systems with 7 edges on 9 vertices
    # We'll verify each is not 2-colorable and is 3-colorable
    
    # Candidate 1: Fano plane on vertices 0-6, vertices 7,8 isolated
    # Fano plane edges (on 7 vertices):
    # 012, 034, 056, 135, 146, 236, 245
    fano_edges = [
        (0,1,2), (0,3,4), (0,5,6),
        (1,3,5), (1,4,6), (2,3,6), (2,4,5)
    ]
    
    # Candidate 2: Another 7-edge construction
    # Using a different 3-uniform 3-chromatic hypergraph
    # This is based on the idea that we need edges that force 3 colors
    
    def is_2_colorable(edges, n=9):
        """Check if hypergraph is 2-colorable by exhaustive search"""
        from itertools import product
        
        # Try all 2-colorings (2^9 = 512 possibilities)
        for bits in product([0,1], repeat=n):
            valid = True
            for (a,b,c) in edges:
                # Edge is monochromatic if all three vertices have same color
                if bits[a] == bits[b] == bits[c]:
                    valid = False
                    break
            if valid:
                return True  # Found a valid 2-coloring
        return False  # No 2-coloring found -> not 2-colorable
    
    def is_3_colorable(edges, n=9):
        """Check if hypergraph is 3-colorable"""
        from itertools import product
        
        # Try all 3-colorings (3^9 = 19683 possibilities)
        for bits in product([0,1,2], repeat=n):
            valid = True
            for (a,b,c) in edges:
                if bits[a] == bits[b] == bits[c]:
                    valid = False
                    break
            if valid:
                return True
        return False
    
    # Test candidate 1
    if not is_2_colorable(fano_edges) and is_3_colorable(fano_edges):
        edges = [list(e) for e in fano_edges]
        # Need to find a valid 3-coloring
        from itertools import product
        for bits in product([0,1,2], repeat=9):
            valid = True
            for (a,b,c) in fano_edges:
                if bits[a] == bits[b] == bits[c]:
                    valid = False
                    break
            if valid:
                colors = [int(b) for b in bits]
                break
        
        # Ensure types are correct
        result = {
            'edges': edges,
            'colors': colors
        }
        
        # Final validation
        assert len(result['edges']) >= 1
        assert all(isinstance(e, list) and len(e) == 3 for e in result['edges'])
        assert all(all(isinstance(v, int) and not isinstance(v, bool) for v in e) for e in result['edges'])
        assert all(0 <= v <= 8 for e in result['edges'] for v in e)
        assert len(result['colors']) == 9
        assert all(isinstance(c, int) and not isinstance(c, bool) for c in result['colors'])
        assert all(c in [0,1,2] for c in result['colors'])
        
        # Verify non-monochromatic edges
        for e in result['edges']:
            a, b, c = e
            assert not (result['colors'][a] == result['colors'][b] == result['colors'][c])
        
        return result
    
    # Fallback: 8-edge construction if 7-edge fails (shouldn't happen)
    # But for safety, return a known valid 8-edge config
    # This is a known 3-chromatic 3-uniform hypergraph on 9 vertices
    fallback_edges = [
        (0,1,2), (0,3,4), (0,5,6),
        (1,3,5), (1,4,6), (2,3,6),
        (2,4,7), (5,6,8)
    ]
    
    edges = [list(e) for e in fallback_edges]
    from itertools import product
    colors = [0]*9
    for bits in product([0,1,2], repeat=9):
        valid = True
        for (a,b,c) in fallback_edges:
            if bits[a] == bits[b] == bits[c]:
                valid = False
                break
        if valid:
            colors = [int(b) for b in bits]
            break
    
    result = {
        'edges': edges,
        'colors': colors
    }
    
    assert len(result['edges']) >= 1
    assert all(isinstance(e, list) and len(e) == 3 for e in result['edges'])
    assert all(all(isinstance(v, int) and not isinstance(v, bool) for v in e) for e in result['edges'])
    assert all(0 <= v <= 8 for e in result['edges'] for v in e)
    assert len(result['colors']) == 9
    assert all(isinstance(c, int) and not isinstance(c, bool) for c in result['colors'])
    assert all(c in [0,1,2] for c in result['colors'])
    
    for e in result['edges']:
        a, b, c = e
        assert not (result['colors'][a] == result['colors'][b] == result['colors'][c])
    
    return result
# EVOLVE-BLOCK-END
