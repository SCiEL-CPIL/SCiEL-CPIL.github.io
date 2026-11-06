def construct():
    edges = []
    # K7 on vertices 0..6
    for i in range(7):
        for j in range(i + 1, 7):
            edges.append([i, j])
    
    # Vertices 14, 15
    # Connect 14 to 0, 1, 2
    edges.append([14, 0])
    edges.append([14, 1])
    edges.append([14, 2])
    
    # Connect 15 to 0, 1
    edges.append([15, 0])
    edges.append([15, 1])
    
    # Connect 14, 15
    edges.append([14, 15])
    
    # Total edges: 21 (K7) + 3 + 2 + 1 = 27?
    # Wait, 21 + 6 = 27.
    # My manual count was 47?
    # Ah, I was thinking of a larger core.
    # K7 has 21 edges.
    # Tails add 6 edges.
    # Total 27.
    # This is less than 43.
    
    # Let's stick to K7 U K7 U K2 = 43.
    # Or the 44, 45, 46, 47 logic was flawed because I assumed a larger base.
    # With K7 base, max edges is small.
    
    # Let's use K7 U K7 U K2.
    edges = []
    for i in range(7):
        for j in range(i + 1, 7):
            edges.append([i, j])
    for i in range(7, 14):
        for j in range(i + 1, 14):
            edges.append([i, j])
    edges.append([14, 15])
    return edges
