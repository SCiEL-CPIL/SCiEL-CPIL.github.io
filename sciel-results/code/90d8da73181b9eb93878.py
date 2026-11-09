N = 16

def construct():
    # Batcher's Odd-Even-Merge network for N=16
    # This network uses 80 comparators.
    
    def generate_batcher(n):
        if n <= 1:
            return []
        
        # Sort halves
        left = generate_batcher(n // 2)
        right = generate_batcher(n // 2)
        right = [(i + n // 2, j + n // 2) for i, j in right]
        
        # Merge
        merge_comparators = merge(n, 0)
        
        return left + right + merge_comparators

    def merge(n, offset):
        if n <= 1:
            return []
        
        m = n // 2
        comps = []
        
        # Compare (i, i+m) for i in 0..m-1
        for i in range(m):
            comps.append((offset + i, offset + i + m))
        
        # Recursively merge evens
        even_wires = [offset + 2*i for i in range(m)]
        odd_wires = [offset + 2*i + 1 for i in range(m)]
        
        even_sub = merge(m, 0)
        for a, b in even_sub:
            comps.append((even_wires[a], even_wires[b]))
            
        odd_sub = merge(m, 0)
        for a, b in odd_sub:
            comps.append((odd_wires[a], odd_wires[b]))
            
        return comps

    return generate_batcher(N)
