import math

def construct():
    N = 17
    comparators = []
    
    def batcher(n):
        if n == 1:
            return []
        n1 = n // 2
        n2 = n - n1
        left = batcher(n1)
        right = batcher(n2)
        
        # Shift right comparators
        for i, j in right:
            right.append((i + n1, j + n1))
        # Wait, the recursive call returns comparators for 0..n2-1.
        # I need to add n1 to them.
        
        # Actually, let's just generate the merge comparators.
        merge = []
        # Stages
        # j is the stage index, 1-based
        # Distance is 2^(j-1)
        # For each i, if i % 2^j < 2^(j-1), compare (i, i + 2^(j-1))
        
        max_stage = math.ceil(math.log2(n))
        for j in range(1, max_stage + 1):
            dist = 1 << (j - 1)
            period = 1 << j
            for i in range(0, n - dist):
                if i % period < dist:
                    merge.append((i, i + dist))
                    
        return left + right_shifted + merge

    # The above logic is slightly flawed in my head.
    # Let's do it cleanly.
    
    def gen_batcher(n, offset=0):
        if n == 1:
            return []
        n1 = n // 2
        n2 = n - n1
        
        left = gen_batcher(n1, offset)
        right = gen_batcher(n2, offset + n1)
        
        merge = []
        max_stage = math.ceil(math.log2(n))
        for j in range(1, max_stage + 1):
            dist = 1 << (j - 1)
            period = 1 << j
            for i in range(0, n - dist):
                if i % period < dist:
                    merge.append((offset + i, offset + i + dist))
                    
        return left + right + merge

    return gen_batcher(N)
