import heapq

def construct():
    """
    Returns the first 20 7-smooth numbers (Hamming numbers with primes 2,3,5,7).
    These numbers have a very small product set due to many multiplicative collisions
    and a small sumset due to their density in the small integer range.
    """
    smooths = set()
    heap = [1]
    smooths_list = []
    primes = [2, 3, 5, 7]
    
    while len(smooths_list) < 20:
        x = heapq.heappop(heap)
        if x in smooths:
            continue
        smooths.add(x)
        smooths_list.append(x)
        
        for p in primes:
            heapq.heappush(heap, x * p)
            
    return smooths_list
