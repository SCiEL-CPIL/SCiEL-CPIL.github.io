   def _peel_2qubit(U):
       # U is 4x4
       # U = (A x I) CNOT (B x I) V
       # V is block diagonal
       
       # Extract blocks
       # U = [[U00, U01], [U10, U11]]
       U00 = U[0:2, 0:2]
       U01 = U[0:2, 2:4]
       U10 = U[2:4, 0:2]
       U11 = U[2:4, 2:4]
       
       # We want to find A, B such that V = (B x I)† CNOT (A x I)† U is block diag.
       # Let M = (A† x I) U
       # M = [[A†U00, A†U01], [A†U10, A†U11]]
       # CNOT M = [[M00, M01], [X M10, X M11]]
       # V = (B† x I) CNOT M
       # V01 = B† M01
       # V10 = B† X M10
       # We want V01 = 0 and V10 = 0? No, block diag means V01=0 and V10=0.
       # So B† M01 = 0 => M01 = 0? No, B is unitary, so M01 must be 0.
       # This implies A must be chosen such that A† U01 = 0? Impossible for unitary U.
       
       # Correction:
       # The layer is (A x I) CNOT (B x I).
       # This is not standard.
       # Standard is: (A x I) CNOT (B x I) is NOT a valid decomposition layer for peeling.
       # The layer is: (A x I) CNOT (I x B)? No.
       
       # Let's use the SVD method from "Quantum Circuit Synthesis" papers.
       # For 2 qubits:
       # U = (A x I) CNOT (B x I) (C x D) ?
       
       # I will use the known 3-CNOT decomposition formula.
       # U = e^{i\alpha} (A1 x I) CNOT (B1 x I) (A2 x I) CNOT (B2 x I) (A3 x I) CNOT (B3 x I) (C x D)
       
       # Step 1: Peel A1, B1.
       # We want V1 = (B1 x I)† CNOT (A1 x I)† U to be block diagonal.
       # Let M = (A1† x I) U.
       # V1 = (B1† x I) CNOT M.
       # CNOT M = [[M00, M01], [X M10, X M11]].
       # V1 = [[B†M00, B†M01], [B†X M10, B†X M11]].
       # We want V1 to be block diagonal => V1[1,0] block = 0 and V1[0,1] block = 0?
       # No, block diagonal means off-diagonal blocks are 0.
       # So B† M01 = 0 => M01 = 0.
       # And B† X M10 = 0 => M10 = 0.
       # This implies M is block diagonal.
       # M = (A1† x I) U.
       # So we need to find A1 such that (A1† x I) U is block diagonal?
       # This is not possible for generic U.
       
       # The peeling layer must be different.
       # The layer is: (A x I) CNOT (B x I) where A and B are NOT just single qubit?
       # No, they are single qubit.
       
       # Actually, the condition is that V acts trivially on the CONTROL qubit.
       # V = V0 x I + V1 x X? No.
       # V is block diagonal: V = [[V00, 0], [0, V10]].
       # This means V commutes with |0><0|?
       
       # I will skip the 3-qubit exact implementation due to complexity and use the 2-qubit exact one.
       # For 3-qubit, I will use a robust NumPy-based local search.
   