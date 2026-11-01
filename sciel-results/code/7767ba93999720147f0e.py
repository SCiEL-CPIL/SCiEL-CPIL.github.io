import numpy as np

# Test Strassen
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C_expected = A @ B  # [[19,22],[43,50]]

# Try the standard formulation
m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
m2 = (A[1,0]+A[1,1])*B[0,0]
m3 = A[0,0]*(B[0,1]-B[1,1])
m4 = A[1,1]*(B[1,0]-B[0,0])
m5 = (A[0,0]+A[0,1])*B[1,1]
m6 = (A[1,0]-A[0,0])*(B[0,0]+B[1,0])
m7 = (A[0,1]-A[1,1])*(B[0,1]+B[1,1])

c00 = m1 + m4 - m5 + m7
c01 = m3 + m5
c10 = m2 + m4
c11 = m1 - m2 + m3 + m6

print(f"c00={c00}, expected {C_expected[0,0]}")
print(f"c01={c01}, expected {C_expected[0,1]}")
print(f"c10={c10}, expected {C_expected[1,0]}")
print(f"c11={c11}, expected {C_expected[1,1]}")
