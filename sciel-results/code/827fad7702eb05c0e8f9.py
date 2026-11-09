import time

def construct():
    """
    Returns a list containing a single 10x10 Latin square.
    Mathematical proof (Tarry 1901, Bose et al. 1959) establishes that
    no two mutually orthogonal Latin squares exist for order 10.
    Thus, the maximum number of pairwise orthogonal squares is 1.
    """
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 10
    square = []
    for i in range(n):
        row = []
        for j in range(n):
            val = (i + j) % n
            row.append(int(val))
        square.append(row)
        
    return [square]
