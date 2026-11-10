def construct():
    from math import gcd
    
    def lcm(a, b):
        return a * b // gcd(a, b)

    # Initialize with a good seed
    best_set = list(range(1, 23))
    
    # Try to improve by adding numbers
    improved = True
    while improved:
        improved = False
        for i in range(1, 513):
            if i in best_set:
                continue
            # Check if i is compatible with all in best_set
            compatible = True
            for j in best_set:
                if lcm(i, j) > 512:
                    compatible = False
                    break
            if compatible:
                best_set.append(i)
                improved = True
                # Sort to keep it neat, though not strictly necessary
                best_set.sort()
                
    # Try to swap elements to increase size?
    # Given the density, simple addition might be enough if we start with a good seed.
    # However, the seed 1..22 is quite good.
    
    return best_set
