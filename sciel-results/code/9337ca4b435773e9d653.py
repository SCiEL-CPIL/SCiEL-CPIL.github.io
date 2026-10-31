import math
import time

def construct():
    # Optimal Golomb rulers for n=1 to n=20
    rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 25], # Check if this is optimal?
        # Actually, let's just use a few known ones and calculate score.
    }
    
    # Known optimal lengths:
    # n=1: 0
    # n=2: 1
    # n=3: 3
    # n=4: 6
    # n=5: 11
    # n=6: 17
    # n=7: 25
    # n=8: 34
    # n=9: 44
    # n=10: 55
    # n=11: 67
    # n=12: 80
    # n=13: 94
    # n=14: 109
    # n=15: 126
    # n=16: 144
    # n=17: 163
    # n=18: 183
    # n=19: 204
    # n=20: 226
    
    # I will hardcode the actual sets for n up to 10 or so, and for larger n, 
    # I'll just return the one for n=10 or 20 if I have it.
    # For simplicity and robustness, I'll use the ones I'm sure of.
    
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25], # This might not be optimal, but it's a Golomb ruler?
        # 25-17=8, 25-12=13, 25-10=15, 25-4=21, 25-1=24, 25-0=25.
        # Previous diffs: 1,3,4,6,9,10,11,12,14,16,17,18,19,20,21?
        # Let's not guess. I'll stick to n<=6 which I verified.
    ]
    
    best_set = [0, 1, 3]
    best_score = 0.0
    
    for u in candidates:
        if 0 not in u:
            u = [0] + u
        u = sorted(set(u))
        
        # Calculate score
        max_u = max(u)
        m = 2 * max_u + 1
        
        # Sumset
        sumset = set()
        for a in u:
            for b in u:
                sumset.add(a + b)
        
        # Diffset
        diffset = set()
        for a in u:
            for b in u:
                diffset.add(a - b)
        
        r = len(diffset) / len(sumset)
        
        if m <= 1:
            score = 1.0
        else:
            score = 1 + math.log(r) / math.log(m)
        
        if score > best_score:
            best_score = score
            best_set = u
            
    return best_set
