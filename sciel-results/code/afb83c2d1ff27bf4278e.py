def priority(vector, n):
    # Use a simple but effective hash to randomize the order
    # This breaks the lexicographic structure which is suboptimal
    h = 0
    for i in range(n):
        h = (h * 31 + vector[i]) % 1000000007
    return h / 1000000007.0
