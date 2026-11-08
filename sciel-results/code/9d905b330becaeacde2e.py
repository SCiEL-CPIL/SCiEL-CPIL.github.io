A = [A1 A2] (2x2 each)
B = [B1; B2] (2x7 each)
C = A1*B1 + A2*B2

For A1*B1 (2x2 * 2x7):
  Split B1 into [B1a B1b] (2x4 and 2x3)
  A1*B1 = A1*B1a + A1*B1b
  
  For A1*B1a (2x2 * 2x4):
    Split B1a into [B1a1 B1a2] (2x2 each)
    A1*B1a = A1*B1a1 + A1*B1a2
    Each 2x2*2x2 uses Strassen (rank 7)
    So A1*B1a uses rank 14
    
  For A1*B1b (2x2 * 2x3):
    Schoolbook: rank 12

So A1*B1 uses rank 14+12 = 26
Similarly A2*B2 uses rank 26
Total: 52
