import time


def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit

    # The optimal set for N=255 containing N is the set of all multiples
    # of the smallest prime factor of N, which is 3.
    # Size is floor(255/3) = 85.
    # Any other construction is smaller or equal.
    
    # Check if we have time (we always do for this simple construction)
    if time.time() > deadline:
        return [255]

    return list(range(3, 256, 3))
