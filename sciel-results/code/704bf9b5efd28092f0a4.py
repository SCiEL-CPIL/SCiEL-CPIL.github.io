def construct():
    n, m, p = 2, 5, 5
    # Partition: A (2x5) = [A1(2x2) | A2(2x2) | A3(2x1)]
    # B (5x5) = [B1(2x5); B2(2x5); B3(1x5)]
    # C = A1*B1 + A2*B2 + A3*B3
    
    # For A1*B1 (2x2 * 2x5):
    #   Split B1 into [B1a(2x2) | B1b(2x2) | B1c(2x1)]
    #   A1*B1a: Strassen 2x2x2 (rank 7)
    #   A1*B1b: Strassen 2x2x2 (rank 7)
    #   A1*B1c: schoolbook 2x2x1 (rank 4)
    
    # Similarly for A2*B2
    
    # A3*B3: schoolbook 2x1 * 1x5 (rank 10)
    
    # Total: 18 + 18 + 10 = 46
    
    # Let me build this explicitly
    ...
