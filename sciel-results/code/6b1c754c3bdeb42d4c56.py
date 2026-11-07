import numpy as np

def update(state, item):
    # Interpret state as 4x128 uint16 array
    # 4 * 128 * 2 = 1024 bytes
    arr = np.frombuffer(state, dtype=np.uint16).reshape(4, 128).copy()
    
    primes = (1035493289, 775374203, 1495862053, 2015145641)
    
    for r in range(4):
        # Calculate index
        # item is int
        idx = ((item * primes[r] + 31 * r) % 2147483647) % 128
        arr[r, idx] += 1
        
    return arr.tobytes()

def query(state, item):
    arr = np.frombuffer(state, dtype=np.uint16).reshape(4, 128)
    
    primes = (1035493289, 775374203, 1495862053, 2015145641)
    
    min_val = 65535 # Max uint16
    
    for r in range(4):
        idx = ((item * primes[r] + 31 * r) % 2147483647) % 128
        val = arr[r, idx]
        if val < min_val:
            min_val = val
            
    return min_val
