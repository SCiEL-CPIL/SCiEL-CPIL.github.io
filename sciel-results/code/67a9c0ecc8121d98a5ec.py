import time

N = 16

def construct():
    """
    Construct a sorting network for 16 wires.
    Returns an ordered list of (i, j) comparators.
    """
    
    def _batcher(start, end):
        length = end - start
        if length <= 1:
            return []
        
        mid = start + length // 2
        comps = []
        comps.extend(_batcher(start, mid))
        comps.extend(_batcher(mid, end))
        comps.extend(_merge(start, mid, length // 2))
        return comps
    
    def _merge(s1, s2, k):
        """
        Merge two sorted sequences of length k.
        First: wires s1..s1+k-1, Second: wires s2..s2+k-1.
        Uses Batcher's odd-even merge.
        """
        if k == 1:
            return [(s1, s2)]
        
        half = k // 2
        comps = []
        
        # Helper to merge two lists of wire indices
        def _merge_idx(a, b, k):
            if k == 1:
                return [(a[0], b[0])]
            
            half = k // 2
            a_even = a[0::2]
            a_odd = a[1::2]
            b_even = b[0::2]
            b_odd = b[1::2]
            
            comps = []
            comps.extend(_merge_idx(a_even, b_even, half))
            comps.extend(_merge_idx(a_odd, b_odd, half))
            
            # Final comparison step
            for i in range(k):
                x, y = a[i], b[i]
                if x < y:
                    comps.append((x, y))
                else:
                    comps.append((y, x))
            
            return comps
        
        a = list(range(s1, s1 + k))
        b = list(range(s2, s2 + k))
        return _merge_idx(a, b, k)
    
    return _batcher(0, N)
