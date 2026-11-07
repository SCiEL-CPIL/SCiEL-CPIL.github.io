import time

def construct():
    # Base set: [65, 128]
    S = set(range(65, 129))
    
    # Add 64
    S.add(64)
    
    # Add odds 1..31
    for i in range(1, 32, 2):
        S.add(i)
        
    # Add 33
    S.add(33)
    
    # Verification logic (optional, for safety)
    # The set is:
    # {1, 3, ..., 31, 33, 64} U {65, ..., 128}
    
    return list(S)
