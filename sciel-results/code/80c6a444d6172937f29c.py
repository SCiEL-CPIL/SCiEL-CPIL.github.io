def construct():
    n = 18
    # Partition vertices into 3 parts of equal size
    part_size = n // 3
    parts = [
        list(range(0, part_size)),
        list(range(part_size, 2 * part_size)),
        list(range(2 * part_size, 3 * part_size))
    ]
    
    edges = []
    # Add edges between all vertices in different parts
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.append([u, v])
                    
    return edges
